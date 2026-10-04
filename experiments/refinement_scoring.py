"""SQL/domain validation and benchmark scoring shared by the model runners."""
import math
import re


def predicate_spec(pred):
    from functionality.predicate import NumericalPredicate
    if isinstance(pred, NumericalPredicate):
        return {"id": pred.get_id(), "expression": pred.attribute.name,
                "operator": pred.operator, "original_value": pred.value,
                "domain": pred.get_valid_value_range()}
    # The original category is always permitted, even when absent from the list.
    return {"id": pred.get_id(), "expression": pred.attribute.name, "operator": "IN",
            "original_value": list(pred.values),
            "domain": {"categories": list(dict.fromkeys(list(pred.values) + list(pred.attribute.categories)))}}



def validate_sql(query, task):
    """Check SQL shape and domains, returning values extracted from SQL itself."""
    import sqlglot
    from sqlglot import exp

    specs = [predicate_spec(p) for p in task.refineable_predicates]
    op_types = {exp.GT: ">", exp.GTE: ">=", exp.LT: "<", exp.LTE: "<=", exp.EQ: "="}

    def key(expr):
        return re.sub(r'[\s"`]', '', expr.sql(dialect="duckdb")).lower()

    def literal(node):
        if isinstance(node, exp.Neg):
            return -float(literal(node.this))
        if not isinstance(node, exp.Literal):
            raise ValueError("Predicate value must be a literal")
        return node.this if node.is_string else float(node.this)

    def mask(sql):
        parsed = sqlglot.parse(sql, read="duckdb")
        if len(parsed) != 1 or not isinstance(parsed[0], exp.Select):
            raise ValueError("Expected exactly one SELECT statement")
        tree = parsed[0]
        values = {}
        for clause_name in ("where", "having"):
            clause = tree.args.get(clause_name)
            if clause is None:
                continue
            for node in list(clause.walk()):
                if type(node) not in op_types and not isinstance(node, exp.In):
                    continue
                for spec in specs:
                    if key(node.this) != re.sub(r'[\s"`]', '', spec['expression']).lower():
                        continue
                    categorical = spec['operator'] == 'IN'
                    if categorical and isinstance(node, (exp.In, exp.EQ)):
                        val = ([literal(v) for v in node.expressions] if isinstance(node, exp.In)
                               else [literal(node.expression)])
                        masked = exp.In(this=node.this.copy(), expressions=[exp.Placeholder()])
                    elif not categorical and op_types.get(type(node)) == spec['operator']:
                        val = literal(node.expression)
                        masked = node.copy()
                        masked.set('expression', exp.Placeholder())
                    else:
                        continue
                    if spec['id'] in values:
                        raise ValueError(f"Repeated predicate: {spec['id']}")
                    values[spec['id']] = val
                    node.replace(masked)
                    break
        for ident in tree.find_all(exp.Identifier):
            ident.set('this', ident.this.lower())
            ident.set('quoted', False)
        return tree.sql(dialect="duckdb"), values

    original_shape, _ = mask(task.original_query)
    candidate_shape, values = mask(query)
    if candidate_shape != original_shape or set(values) != {s['id'] for s in specs}:
        raise ValueError("SQL changes structure, operators, or the allowed predicate set")
    issues = []
    for spec in specs:
        val, domain = values[spec['id']], spec['domain']
        if spec['operator'] == 'IN':
            valid = bool(val) and set(val).issubset(domain['categories'])
        elif 'valid_values' in domain:
            valid = any(math.isclose(val, float(v), rel_tol=1e-9, abs_tol=1e-8)
                        for v in domain['valid_values'])
        else:
            step = float(domain['step'])
            offset = (val - float(domain['min'])) / step if step else 0
            valid = (float(domain['min']) - 1e-8 <= val <= float(domain['max']) + 1e-8
                     and math.isclose(offset, round(offset), abs_tol=1e-6))
        # Keeping an original value is a valid no-op even if a legacy domain omits it.
        if val == spec['original_value']:
            valid = True
        if not valid:
            issues.append(spec['id'])
    return values, issues



def score(query, task, config, index):
    import duckdb
    values, domain_issues = validate_sql(query, task)
    con = duckdb.connect(config={"enable_external_access": False, "threads": 1})
    try:
        for name, df in (task.df if isinstance(task.df, dict) else {'df': task.df}).items():
            con.register(name, df)
        result = con.execute(query).df()
    finally:
        con.close()
    deviations = [float(c.evaluate(result)) for c in task.output_constraints]
    constraint_values = [float(c.query(result) if hasattr(c, 'query') else c.get_query_value(result))
                         for c in task.output_constraints]
    csat = sum(deviations) / len(deviations)
    distance = float(task.refinement_objective(query))
    satisfied = bool(len(result) and math.isfinite(csat) and csat <= config['epsilon'])
    success = satisfied and math.isfinite(distance) and not domain_issues
    raw_optimality = ((float(config['max_dist'].iloc[index]) - distance) /
                      (float(config['max_dist'].iloc[index]) - float(config['gt'].iloc[index]) + 1e-9))
    return {"success": success, "constraints_satisfied": satisfied,
            "constraint_score": csat, "distance": distance,
            "optimality": max(0., min(1., raw_optimality)) if success else 0.,
            "raw_optimality": raw_optimality, "result_rows": len(result),
            "constraint_values": constraint_values, "constraint_deviations": deviations,
            "predicate_values": values, "domain_issues": domain_issues}
