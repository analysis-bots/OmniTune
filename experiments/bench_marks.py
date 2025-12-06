import math
from enum import Enum
import pandas as pd

# from config import OUT_PATH, ACTOR_ONLY, CRITIC_ONLY_RANDOM
from opro.opro_main_loop import run_task
from functionality.constraint import DiversityCardinalityConstraint, DiverseTopKSelectionConstraint, \
    RangeQueryFairnessConstraint, AgnosticConstraint
from functionality.objectives import get_script_diff_func_sql, get_having_predicate_distance_function
from functionality.predicate import NumericalAttribute, CategoricalAttribute, NumericalPredicate, CategoricalPredicate
from functionality.task import DiversityConstraintsTask, TopKRefinementTask, RangeQueryRefinementTask, AgnosticTask
from tools.utils import construct_predicate_steps_by_jaccard


class TaskType(Enum):
    TOP_K_REFINEMENT = 0
    RANGE_QUERY_REFINEMENT = 1
    PROVENANCE = 2

### TOP K REFINEMENT ###

## Astronauts Dataset ##

df1 = pd.read_csv("datasets/top_k_refinement/astronauts_500kb.csv")

# Task 1a #

T1a_ORIGINAL_SCRIPT_SQL = """
SELECT * FROM df
WHERE "Graduate Major" = 'Aeronautics & Astronautics' AND "Space Walks" >= 8 AND "Space Walks" <= 9
ORDER BY "Space Flight (hr)" DESC;
"""

c1a1 = DiverseTopKSelectionConstraint(k=10, sign=1, desired_value=5, attribute='Gender', operator='=', identifier='Female')
c1a2 = DiverseTopKSelectionConstraint(k=10, sign=1, desired_value=2, attribute='Status', operator='=', identifier='Active')
c1a3 = DiverseTopKSelectionConstraint(k=10, sign=1, desired_value=2, attribute='Status', operator='=', identifier='Management')
T1a_CONSTRAINTS = [c1a1, c1a2, c1a3]
T1a_refineable_attributes = [NumericalAttribute(name="Space Walks", min_value=0, max_value=10, step=1),
                            CategoricalAttribute(name="Graduate Major", categories=['Aeronautics & Astronautics'] + df1['Graduate Major'].value_counts().head(5).index.tolist())]

T1a_refineable_predicates = [NumericalPredicate(T1a_refineable_attributes[0], None, ">=", 8),
                             NumericalPredicate(T1a_refineable_attributes[0], None, "<=", 9),
                             CategoricalPredicate(T1a_refineable_attributes[1], None, ["Aeronautics & Astronautics"])]


T1a = TopKRefinementTask(name="Astronauts1", df=df1, constraints=T1a_CONSTRAINTS, query=T1a_ORIGINAL_SCRIPT_SQL,
                         refineable_predicates=T1a_refineable_predicates)

# Task 1b #
T1b_ORIGINAL_SCRIPT_SQL = """
SELECT * FROM df
WHERE "Graduate Major" IN ('Aerospace Engineering') AND "Space Walks" >= 2 AND "Space Walks" <= 5
ORDER BY "Space Flight (hr)" DESC;
"""

c1b1 = DiverseTopKSelectionConstraint(k=10, sign=1, desired_value=3, attribute='Status', operator='=', identifier='Management')
c1b2 = DiverseTopKSelectionConstraint(k=10, sign=1, desired_value=5, attribute='Gender', operator='=', identifier='Female')
T1b_CONSTRAINTS = [c1b1, c1b2]

T1b_refineable_attributes = [NumericalAttribute(name="Space Walks", min_value=0, max_value=10, step=1),
                            CategoricalAttribute(name="Graduate Major", categories=['Astronautics'] + df1['Graduate Major'].value_counts().head(3).index.tolist())]

T1b_refineable_predicates = [NumericalPredicate(T1b_refineable_attributes[0], None, ">=", 2),
                             NumericalPredicate(T1b_refineable_attributes[0], None, "<=", 5),
                             CategoricalPredicate(T1b_refineable_attributes[1], None, ["Aerospace Engineering"])]

T1b = TopKRefinementTask(name="Astronauts2", df=df1, constraints=T1b_CONSTRAINTS, query=T1b_ORIGINAL_SCRIPT_SQL,
                         refineable_predicates=T1b_refineable_predicates)
## Law Students Dataset ##

df2 = pd.read_csv("datasets/top_k_refinement/law_students.csv")

# Task 2a #

T2a_ORIGINAL_SCRIPT_SQL = """
SELECT * FROM df
WHERE "region_first" = 'PO' AND "UGPA" >= 3.0 AND "UGPA" <= 3.5
ORDER BY "LSAT" DESC;
"""

c2a1 = DiverseTopKSelectionConstraint(k=10, sign=1, desired_value=5, attribute='gender', operator='=', identifier='F')
c2a2 = DiverseTopKSelectionConstraint(k=10, sign=1, desired_value=2, attribute='ethnicity', operator='=', identifier='Hispanic')
T2a_CONSTRAINTS = [c2a1, c2a2]

T2a_refineable_attributes = [NumericalAttribute(name="UGPA", min_value=0.0, max_value=4.0, step=0.1),
                            CategoricalAttribute(name="region_first", categories=['PO'] + df2['region_first'].value_counts().head(4).index.tolist())]
T2a_refineable_predicates = [NumericalPredicate(T2a_refineable_attributes[0], None, ">=", 3.0),
                                NumericalPredicate(T2a_refineable_attributes[0], None, "<=", 3.5),
                                CategoricalPredicate(T2a_refineable_attributes[1], None, ["PO"])]

T2a = TopKRefinementTask(name="Students1", df=df2, constraints=T2a_CONSTRAINTS, query=T2a_ORIGINAL_SCRIPT_SQL,
                         refineable_predicates=T2a_refineable_predicates)

# Task 2b #

T2b_ORIGINAL_SCRIPT_SQL = """
SELECT * FROM df
WHERE "region_first" = 'Mt' AND "UGPA" >= 3.3 AND "UGPA" <= 3.9
ORDER BY "LSAT" DESC;
"""

c2b1 = DiverseTopKSelectionConstraint(k=10, sign=1, desired_value=5, attribute='gender', operator='=', identifier='F')
c2b2 = DiverseTopKSelectionConstraint(k=10, sign=1, desired_value=2, attribute='ethnicity', operator='=', identifier='Asian')
T2b_CONSTRAINTS = [c2b1, c2b2]

T2b_refineable_attributes = [NumericalAttribute(name="UGPA", min_value=0.0, max_value=4.0, step=0.1),
                            CategoricalAttribute(name="region_first", categories=df2['region_first'].unique())]
T2b_refineable_predicates = [NumericalPredicate(T2b_refineable_attributes[0], None, ">=", 3.3),
                                NumericalPredicate(T2b_refineable_attributes[0], None, "<=", 3.9),
                                CategoricalPredicate(T2b_refineable_attributes[1], None, ["Mt"])]

T2b = TopKRefinementTask(name="Students2", df=df2, constraints=T2b_CONSTRAINTS, query=T2b_ORIGINAL_SCRIPT_SQL,
                            refineable_predicates=T2b_refineable_predicates)

## MEPS Dataset ##

df3 = pd.read_csv("datasets/top_k_refinement/meps.csv")

# Task 3a #

T3a_ORIGINAL_SCRIPT_SQL = """
SELECT * FROM df
WHERE "AGE16X" >= 12 AND "FAMS1231" >= 5
ORDER BY "OBTOTV16" DESC;
"""

c3a1 = DiverseTopKSelectionConstraint(k=10, sign=1, desired_value=5, attribute='SEX', operator='=', identifier='F')
c3a2 = DiverseTopKSelectionConstraint(k=10, sign=1, desired_value=3, attribute='RACEV1X', operator='=', identifier='Black')
T3a_CONSTRAINTS = [c3a1, c3a2]

T3a_refineable_attributes = [NumericalAttribute(name="AGE16X", min_value=0, max_value=90, step=1),
                            NumericalAttribute(name="FAMS1231", min_value=0, max_value=10, step=1)]
T3a_refineable_predicates = [NumericalPredicate(T3a_refineable_attributes[0], None, ">=", 12),
                             NumericalPredicate(T3a_refineable_attributes[1], None, ">=", 5)]

T3a = TopKRefinementTask(name="MEPS1", df=df3,
                         constraints=T3a_CONSTRAINTS,
                         query=T3a_ORIGINAL_SCRIPT_SQL,
                         refineable_predicates=T3a_refineable_predicates)
# Task 3b #

T3b_ORIGINAL_SCRIPT_SQL = """
SELECT * FROM df
WHERE "AGE16X" >= 52 AND "FAMS1231" >= 4
ORDER BY "OBTOTV16" DESC;
"""

c3b1 = DiverseTopKSelectionConstraint(k=10, sign=1, desired_value=5, attribute='SEX', operator='=', identifier='F')
c3b2 = DiverseTopKSelectionConstraint(k=10, sign=1, desired_value=2, attribute='RACEV1X', operator='=', identifier='Asian')
T3b_CONSTRAINTS = [c3b1, c3b2]

T3b_refineable_attributes = [NumericalAttribute(name="AGE16X", min_value=0, max_value=90, step=1),
                            NumericalAttribute(name="FAMS1231", min_value=0, max_value=10, step=1)]
T3b_refineable_predicates = [NumericalPredicate(T3b_refineable_attributes[0], None, ">=", 52),
                                NumericalPredicate(T3b_refineable_attributes[1], None, ">=", 4)]

T3b = TopKRefinementTask(name="MEPS2", df=df3,
                         constraints=T3b_CONSTRAINTS,
                         query=T3b_ORIGINAL_SCRIPT_SQL,
                         refineable_predicates=T3b_refineable_predicates)


## TPC-H Dataset ##

df4 = pd.read_csv("datasets/top_k_refinement/tcp_h.csv")

# Task 4a #

T4a_ORIGINAL_SCRIPT_SQL = """SELECT * FROM df
WHERE "REGION_NAME" IN ('MIDDLE EAST') AND "QUANTITY" >= 10 AND "QUANTITY" <= 25
ORDER BY "REVENUE1" DESC;"""

c4a1 = DiverseTopKSelectionConstraint(k=10, sign=1, desired_value=5, attribute='ORDERPRIORITY', operator='=', identifier='5-LOW')
c4a2 = DiverseTopKSelectionConstraint(k=10, sign=1, desired_value=2, attribute='MKTSEGMENT', operator='=', identifier='MACHINERY')
T4a_CONSTRAINTS = [c4a1, c4a2]

T4a_refineable_attributes = [NumericalAttribute(name="QUANTITY", min_value=1, max_value=50, step=1),
                            CategoricalAttribute(name="REGION_NAME", categories=df4['REGION_NAME'].unique())]
T4a_refineable_predicates = [NumericalPredicate(T4a_refineable_attributes[0], None, ">=", 10),
                                NumericalPredicate(T4a_refineable_attributes[0], None, "<=", 25),
                                CategoricalPredicate(T4a_refineable_attributes[1], None, ["MIDDLE EAST"])]


T4a = TopKRefinementTask(name="TPC-H1", df=df4,
                        constraints=T4a_CONSTRAINTS,
                        query=T4a_ORIGINAL_SCRIPT_SQL,
                        refineable_predicates=T4a_refineable_predicates)

# Task 4b #

T4b_ORIGINAL_SCRIPT_SQL = """SELECT * FROM df
WHERE "REGION_NAME" IN ('ASIA')
AND "QUANTITY" >= 5 AND "QUANTITY" <= 7
ORDER BY "REVENUE1" DESC;"""

c4b1 = DiverseTopKSelectionConstraint(k=10, sign=1, desired_value=5, attribute='ORDERPRIORITY', operator='=', identifier='5-LOW')
c4b2 = DiverseTopKSelectionConstraint(k=10, sign=1, desired_value=2, attribute='MKTSEGMENT', operator='=', identifier='HOUSEHOLD')
T4b_CONSTRAINTS = [c4b1, c4b2]

T4b_refineable_attributes = [NumericalAttribute(name="QUANTITY", min_value=1, max_value=50, step=1),
                            CategoricalAttribute(name="REGION_NAME", categories=df4['REGION_NAME'].unique())]
T4b_refineable_predicates = [NumericalPredicate(T4b_refineable_attributes[0], None, ">=", 5),
                                NumericalPredicate(T4b_refineable_attributes[0], None, "<=", 7),
                                CategoricalPredicate(T4b_refineable_attributes[1], None, ["ASIA"])]

T4b = TopKRefinementTask(name="TPC-H2", df=df4,
                        constraints=T4b_CONSTRAINTS,
                        query=T4b_ORIGINAL_SCRIPT_SQL,
                        refineable_predicates=T4b_refineable_predicates)


### RANGE QUERY REFINEMENT ###

## Texas Tribune Dataset ##

df5 = pd.read_csv("datasets/range_query_refinement/texas_2024.csv")
MAX_VAL5 = df5["ANNUAL"].max() + 5000  # Assuming the maximum value is the highest annual income + step size
# Task 5a #

T5a_ORIGINAL_SCRIPT_SQL = """
SELECT * FROM df
WHERE "ANNUAL" > 50000 AND "ANNUAL" < 125000;
"""

c5a1 = RangeQueryFairnessConstraint(attribute='GENDER', identifier='FEMALE', w_red=4, w_blue=3, desired_value=2000)
T5a_CONSTRAINTS = [c5a1]

T5a_refinaable_attributes = [NumericalAttribute(name="ANNUAL", min_value=0, max_value=MAX_VAL5, step=5000)]

T5aR1 = {"valid_values": construct_predicate_steps_by_jaccard("ANNUAL", ">", orig_val=50000, df=df5)}
T5aR2 = {"valid_values": construct_predicate_steps_by_jaccard("ANNUAL", "<", orig_val=125000, df=df5)}

T5a_refineable_predicates = [NumericalPredicate(T5a_refinaable_attributes[0], T5aR1, ">", 50000),
                                NumericalPredicate(T5a_refinaable_attributes[0], T5aR2, "<", 125000)]



T5a = RangeQueryRefinementTask(name="Texas Tribune1", df=df5, constraints=T5a_CONSTRAINTS,
                               query=T5a_ORIGINAL_SCRIPT_SQL, refineable_predicates=T5a_refineable_predicates)

# Task 5b #

T5b_ORIGINAL_SCRIPT_SQL = """
SELECT * FROM df
WHERE "ANNUAL" > 85000 AND "ANNUAL" < 140000;
"""

c5b1 = RangeQueryFairnessConstraint(attribute='ETHNICITY', identifier='WHITE', w_red=3, w_blue=2, desired_value=1000)
T5b_CONSTRAINTS = [c5b1]

T5b_refineable_attributes = [NumericalAttribute(name="ANNUAL", min_value=0, max_value=MAX_VAL5, step=5000)]

T5bR1 = {"valid_values": construct_predicate_steps_by_jaccard("ANNUAL", ">", orig_val=85000, df=df5)}
T5bR2 = {"valid_values": construct_predicate_steps_by_jaccard("ANNUAL", "<", orig_val=140000, df=df5)}

T5b_refineable_predicates = [NumericalPredicate(T5b_refineable_attributes[0], T5bR1, ">", 85000),
                                NumericalPredicate(T5b_refineable_attributes[0], T5bR2, "<", 140000)]

T5b = RangeQueryRefinementTask(name="Texas Tribune2", df=df5, constraints=T5b_CONSTRAINTS,
                               query=T5b_ORIGINAL_SCRIPT_SQL, refineable_predicates=T5b_refineable_predicates)


## COMPASS Dataset ##

df6 = pd.read_csv("datasets/range_query_refinement/compas.csv")

# Task 6a #

T6a_ORIGINAL_SCRIPT_SQL = """
SELECT * FROM df
WHERE "RawScore" >= 28 AND "RawScore" <= 45;
"""

c6a1 = RangeQueryFairnessConstraint(attribute='Ethnic_Code_Text', identifier='Caucasian', w_red=2, w_blue=1, desired_value=800)
T6a_CONSTRAINTS = [c6a1]

T6a_refineable_attributes = [NumericalAttribute(name="RawScore", min_value=-4, max_value=51, step=1)]

T6aR1 = {"valid_values": construct_predicate_steps_by_jaccard("RawScore", ">=", orig_val=28, df=df6)}
T6aR2 = {"valid_values": construct_predicate_steps_by_jaccard("RawScore", "<=", orig_val=45, df=df6)}

T6a_refineable_predicates = [NumericalPredicate(T6a_refineable_attributes[0], T6aR1, ">=", 28),
                            NumericalPredicate(T6a_refineable_attributes[0], T6aR2, "<=", 45)]

T6a = RangeQueryRefinementTask(name="COMPASS1", df=df6, constraints=T6a_CONSTRAINTS, query=T6a_ORIGINAL_SCRIPT_SQL,
                                 refineable_predicates=T6a_refineable_predicates)

# Task 6b #

T6b_ORIGINAL_SCRIPT_SQL = """
SELECT * FROM df
WHERE "RawScore" >= 30 AND "RawScore" <= 40;
"""

c6b1 = RangeQueryFairnessConstraint(attribute='Ethnic_Code_Text', identifier='Caucasian', w_red=3, w_blue=2, desired_value=600)
T6b_CONSTRAINTS = [c6b1]

T6a_refineable_attributes = [NumericalAttribute(name="RawScore", min_value=-4, max_value=51, step=1)]

T6bR1 = {"valid_values": construct_predicate_steps_by_jaccard("RawScore", ">=", orig_val=30, df=df6)}
T6bR2 = {"valid_values": construct_predicate_steps_by_jaccard("RawScore", "<=", orig_val=40, df=df6)}

T6b_refineable_predicates = [NumericalPredicate(T6a_refineable_attributes[0], T6bR1, ">=", 30),
                            NumericalPredicate(T6a_refineable_attributes[0], T6bR2, "<=", 40)]

T6b = RangeQueryRefinementTask(name="COMPASS2", df=df6, constraints=T6b_CONSTRAINTS, query=T6b_ORIGINAL_SCRIPT_SQL,
                                    refineable_predicates=T6b_refineable_predicates)

## Housing Prices Dataset ##

df7 = pd.read_csv("datasets/range_query_refinement/housing_prices.csv")


# Task 7a #

T7a_ORIGINAL_SCRIPT_SQL = """
SELECT * FROM df
WHERE "price" >= 5000000 AND "price" <= 6000000
"""

c7a1 = RangeQueryFairnessConstraint(attribute='response', identifier='yes', w_red=3, w_blue=1, desired_value=50)
T7a_CONSTRAINTS = [c7a1]

T7a_refineable_attributes = [NumericalAttribute(name="price", min_value=1800000, max_value=13300000, step=100000)]

T7aR1 = {"valid_values": construct_predicate_steps_by_jaccard("price", ">=", orig_val=5000000, df=df7)}
T7aR2 = {"valid_values": construct_predicate_steps_by_jaccard("price", "<=", orig_val=6000000, df=df7)}

T7a_refineable_predicates = [NumericalPredicate(T7a_refineable_attributes[0], T7aR1, ">=", 5000000),
                                NumericalPredicate(T7a_refineable_attributes[0], T7aR2, "<=", 6000000)]

T7a = RangeQueryRefinementTask(name="Housing Prices1", df=df7, constraints=T7a_CONSTRAINTS,
                               query=T7a_ORIGINAL_SCRIPT_SQL, refineable_predicates=T7a_refineable_predicates)

# Task 7b #

T7b_ORIGINAL_SCRIPT_SQL = """
SELECT * FROM df
WHERE "area" >= 6000 AND "area" <= 10000
"""
c7b1 = RangeQueryFairnessConstraint(attribute='basement', identifier='yes', w_red=2, w_blue=1, desired_value=50)
T7b_CONSTRAINTS = [c7b1]

T7b_refineable_attributes = [NumericalAttribute(name="area", min_value=1650, max_value=16200, step=50)]

T7bR1 = {"valid_values": construct_predicate_steps_by_jaccard("area", ">=", orig_val=6000, df=df7)}
T7bR2 = {"valid_values": construct_predicate_steps_by_jaccard("area", "<=", orig_val=10000, df=df7)}

T7b_refineable_predicates = [NumericalPredicate(T7b_refineable_attributes[0], T7bR1, ">=", 6000),
                            NumericalPredicate(T7b_refineable_attributes[0], T7bR2, "<=", 10000)]

T7b = RangeQueryRefinementTask(name="Housing Prices2", df=df7, constraints=T7b_CONSTRAINTS,
                               query=T7b_ORIGINAL_SCRIPT_SQL,
                               refineable_predicates=T7b_refineable_predicates)


## Fraud Detection Dataset ##

df8 = pd.read_csv("datasets/range_query_refinement/fraud.csv")

# Task 8a #

T8a_ORIGINAL_SCRIPT_SQL = """
SELECT * FROM df
WHERE "Transaction_Amount" >= 450 AND "Transaction_Amount" <= 1000
"""

c8a1 = RangeQueryFairnessConstraint(attribute='Fraud_Label', identifier=1, w_red=3, w_blue=2, desired_value=500)
T8a_CONSTRAINTS = [c8a1]

T8a_refineable_attributes = [NumericalAttribute(name="Transaction_Amount", min_value=0, max_value=1170, step=10)]

T8aR1 = {"valid_values": construct_predicate_steps_by_jaccard("Transaction_Amount", ">=", orig_val=450, df=df8)}
T8aR2 = {"valid_values": construct_predicate_steps_by_jaccard("Transaction_Amount", "<=", orig_val=1000, df=df8)}

T8a_refineable_predicates = [NumericalPredicate(T8a_refineable_attributes[0], T8aR1, ">=", 450),
                            NumericalPredicate(T8a_refineable_attributes[0], T8aR2, "<=", 1000)]

T8a = RangeQueryRefinementTask(name="Fraud Detection1", df=df8, constraints=T8a_CONSTRAINTS,
                               query=T8a_ORIGINAL_SCRIPT_SQL,
                               refineable_predicates=T8a_refineable_predicates)


# Task 8b #

T8b_ORIGINAL_SCRIPT_SQL = """
SELECT * FROM df
WHERE "Transaction_Amount" >= 400 AND "Transaction_Amount" <= 800
"""

c8b1 = RangeQueryFairnessConstraint(attribute='Fraud_Label', identifier=1, w_red=2, w_blue=1, desired_value=500)
T8b_CONSTRAINTS = [c8b1]

T8b_refineable_attributes = [NumericalAttribute(name="Transaction_Amount", min_value=0, max_value=1170, step=10)]

T8bR1 = {"valid_values": construct_predicate_steps_by_jaccard("Transaction_Amount", ">=", orig_val=400, df=df8)}
T8bR2 = {"valid_values": construct_predicate_steps_by_jaccard("Transaction_Amount", "<=", orig_val=800, df=df8)}

T8b_refineable_predicates = [NumericalPredicate(T8b_refineable_attributes[0], T8bR1, ">=", 400),
                            NumericalPredicate(T8b_refineable_attributes[0],T8bR2, "<=", 800)]

T8b = RangeQueryRefinementTask(name="Fraud Detection2", df=df8, constraints=T8b_CONSTRAINTS,
                               query=T8b_ORIGINAL_SCRIPT_SQL,
                               refineable_predicates=T8b_refineable_predicates)

### DIVERSITY CONSTRAINTS SATISFACTION ###


## Healthcare Dataset ##
df9 = pd.read_csv('datasets/diversity_constraint_satisfaction/healthcare.csv')

# Task 9a #
T9a_ORIGINAL_SCRIPT_SQL = """
SELECT * FROM df
WHERE "income" >= 200.0 AND "num_children" >= 3 AND "county" IN ('county2','county3');
"""

c9a1 = DiversityCardinalityConstraint(attribute='race', identifier='race2', symbol='>=', desired_value=20)
c9a2 = DiversityCardinalityConstraint(attribute='all', identifier='yes', symbol='<=', desired_value=100)
T9a_CONSTRAINTS = [c9a1, c9a2]

T9a_refineable_attributes = [NumericalAttribute(name="income", min_value=10, max_value=450, step=10),
                             NumericalAttribute(name="num_children", min_value=1, max_value=5, step=1),
                                CategoricalAttribute(name="county", categories=df9['county'].unique())]
T9a_refineable_predicates = [NumericalPredicate(T9a_refineable_attributes[0], None, ">=", 200),
                                NumericalPredicate(T9a_refineable_attributes[1], None, ">=", 3),
                                CategoricalPredicate(T9a_refineable_attributes[2], None, ["county2", "county3"])]

T9a = DiversityConstraintsTask(name="Healthcare1", df=df9, constraints=T9a_CONSTRAINTS, query=T9a_ORIGINAL_SCRIPT_SQL,
                               max_df_length=100, refineable_predicates=T9a_refineable_predicates)

# Task 9b #
T9b_ORIGINAL_SCRIPT_SQL = """
SELECT * FROM df
WHERE "income" <= 250.0 AND "complications" >= 4 AND "num_children" >= 3;
"""

c9b1 = DiversityCardinalityConstraint(attribute='age_group', identifier='group3', symbol='>=', desired_value=20)
c9b2 = DiversityCardinalityConstraint(attribute='all', identifier='yes', symbol='<=', desired_value=100)
T9b_CONSTRAINTS = [c9b1, c9b2]

T9b_refineable_attributes = [NumericalAttribute(name="income", min_value=10, max_value=450, step=10),
                                NumericalAttribute(name="complications", min_value=0, max_value=10, step=1),
                                NumericalAttribute(name="num_children", min_value=0, max_value=5, step=1)]

T9b_refineable_predicates = [NumericalPredicate(T9b_refineable_attributes[0], None, "<=", 250),
                                NumericalPredicate(T9b_refineable_attributes[1], None, ">=", 4),
                                NumericalPredicate(T9b_refineable_attributes[2], None, ">=", 3)]


T9b = DiversityConstraintsTask(name="Healthcare2", df=df9, constraints=T9b_CONSTRAINTS, query=T9b_ORIGINAL_SCRIPT_SQL,
                               max_df_length=100, refineable_predicates=T9b_refineable_predicates)


## ACS Income Dataset ##

df10 = pd.read_csv('datasets/diversity_constraint_satisfaction/ACSIncome.csv')


# Task 10a #

T10a_ORIGINAL_SCRIPT_SQL = """
SELECT * FROM df
WHERE "hours_per_week" >= 40.0 AND "education_num" >= 10.0 
AND "workclass" IN ('Local_gov');
"""

c10a1 = DiversityCardinalityConstraint(attribute='sex', identifier='Female', symbol='>=', desired_value=75)
c10a2 = DiversityCardinalityConstraint(attribute='all', identifier='yes', symbol='<=', desired_value=250)


T10a_refineable_attributes = [NumericalAttribute(name="hours_per_week", min_value=1, max_value=99, step=1),
                                NumericalAttribute(name="education_num", min_value=1, max_value=16, step=1),
                                CategoricalAttribute(name="workclass", categories=df10['workclass'].unique())]

T10a_refineable_predicates = [NumericalPredicate(T10a_refineable_attributes[0], None, ">=", 40),
                                NumericalPredicate(T10a_refineable_attributes[1], None, ">=", 10),
                                CategoricalPredicate(T10a_refineable_attributes[2], None, ["Local_gov"])]


T10a_CONSTRAINTS = [c10a1, c10a2]
T10a = DiversityConstraintsTask(name="ACSIncome1", df=df10, constraints=T10a_CONSTRAINTS, query=T10a_ORIGINAL_SCRIPT_SQL,
                                max_df_length=1000, refineable_predicates=T10a_refineable_predicates)


# Task 10b #

T10b_ORIGINAL_SCRIPT_SQL = """
SELECT * FROM df
WHERE "hours_per_week" >= 25.0 AND "education_num" >= 12.0 
AND "workclass" IN ('Local_gov','State_gov','Federal_gov');
"""

c10b1 = DiversityCardinalityConstraint(attribute='sex', identifier='Female', symbol='>=', desired_value=300)
c10b2 = DiversityCardinalityConstraint(attribute='race', identifier='Black', symbol='>=', desired_value=100)
c10b3 = DiversityCardinalityConstraint(attribute='all', identifier='yes', symbol='<=', desired_value=1000)
T10b_CONSTRAINTS = [c10b1, c10b2, c10b3]

T10b_refineable_attributes = [NumericalAttribute(name="hours_per_week", min_value=1, max_value=99, step=1),
                                NumericalAttribute(name="education_num", min_value=1, max_value=16, step=1),
                                CategoricalAttribute(name="workclass", categories=df10['workclass'].unique())]
T10b_refineable_predicates = [NumericalPredicate(T10b_refineable_attributes[0], None, ">=", 25),
                                NumericalPredicate(T10b_refineable_attributes[1], None, ">=", 12),
                                CategoricalPredicate(T10b_refineable_attributes[2], None, ["Local_gov", "State_gov", "Federal_gov"])]


T10b = DiversityConstraintsTask(name="ACSIncome2", df=df10, constraints=T10b_CONSTRAINTS, query=T10b_ORIGINAL_SCRIPT_SQL,
                                max_df_length=1000, refineable_predicates=T10b_refineable_predicates)


## COMPAS Dataset ##

df11 = pd.read_csv('datasets/diversity_constraint_satisfaction/compas.csv')

# Task 11a #

T11a_ORIGINAL_SCRIPT_SQL = """
SELECT * FROM df
WHERE "age" >= 40.0 AND "decile_score" >= 5.0 AND "c_charge_degree" IN ('O','M');
"""

c11a1 = DiversityCardinalityConstraint(attribute='sex', identifier='Female', symbol='>=', desired_value=100)
c11a2 = DiversityCardinalityConstraint(attribute='all', identifier='yes', symbol='<=', desired_value=600)
T11a_CONSTRAINTS = [c11a1, c11a2]

T11a_refineable_attributes = [NumericalAttribute(name="age", min_value=18, max_value=96, step=1),
                                NumericalAttribute(name="decile_score", min_value=-1, max_value=10, step=1),
                                CategoricalAttribute(name="c_charge_degree", categories=df11['c_charge_degree'].unique())]
T11a_refineable_predicates = [NumericalPredicate(T11a_refineable_attributes[0], None, ">=", 40),
                                NumericalPredicate(T11a_refineable_attributes[1], None, ">=", 5),
                                CategoricalPredicate(T11a_refineable_attributes[2], None, ["O", "M"])]


T11a = DiversityConstraintsTask(name="COMPAS1", df=df11, constraints=T11a_CONSTRAINTS, query=T11a_ORIGINAL_SCRIPT_SQL,
                                max_df_length=1000, refineable_predicates=T11a_refineable_predicates)

# Task 11b #

T11b_ORIGINAL_SCRIPT_SQL = """
SELECT * FROM df
WHERE "age" >= 38.0 AND "decile_score" >= 4.0 AND "c_charge_degree" IN ('M');
"""

c11b1 = DiversityCardinalityConstraint(attribute='race', identifier='Caucasian', symbol='>=', desired_value=200)
c11b2 = DiversityCardinalityConstraint(attribute='all', identifier='yes', symbol='<=', desired_value=800)
T11b_CONSTRAINTS = [c11b1, c11b2]

T11b_refineable_attributes = [NumericalAttribute(name="age", min_value=18, max_value=96, step=1),
                                NumericalAttribute(name="decile_score", min_value=-1, max_value=10, step=1),
                                CategoricalAttribute(name="c_charge_degree", categories=df11['c_charge_degree'].unique())]

T11b_refineable_predicates = [NumericalPredicate(T11b_refineable_attributes[0], None, ">=", 38),
                                NumericalPredicate(T11b_refineable_attributes[1], None, ">=", 4),
                                CategoricalPredicate(T11b_refineable_attributes[2], None, ["M"])]


T11b = DiversityConstraintsTask(name="COMPAS2", df=df11, constraints=T11b_CONSTRAINTS, query=T11b_ORIGINAL_SCRIPT_SQL,
                                max_df_length=1000, refineable_predicates=T11b_refineable_predicates)

## Students Dataset ##

df12 = pd.read_csv('datasets/diversity_constraint_satisfaction/students.csv')

# Task 12a #

T12a_ORIGINAL_SCRIPT_SQL = """
SELECT * FROM df
WHERE "math_score" >= 50.0 AND "writing_score" >= 60.0 
AND "parental_level_of_education" IN ('masters_degree');
"""

c12a1 = DiversityCardinalityConstraint(attribute='gender', identifier='female', symbol='>=', desired_value=200)
c12a2 = DiversityCardinalityConstraint(attribute='all', identifier='yes', symbol='<=', desired_value=400)
T12a_CONSTRAINTS = [c12a1, c12a2]

T12a_refineable_attributes = [NumericalAttribute(name="math_score", min_value=0, max_value=100, step=5),
                                NumericalAttribute(name="writing_score", min_value=10, max_value=100, step=5),
                                CategoricalAttribute(name="parental_level_of_education", categories=df12['parental_level_of_education'].unique())]

T12a_refineable_predicates = [NumericalPredicate(T12a_refineable_attributes[0], None, ">=", 50),
                                NumericalPredicate(T12a_refineable_attributes[1], None, ">=", 60),
                                CategoricalPredicate(T12a_refineable_attributes[2], None, ["masters_degree"])]

T12a = DiversityConstraintsTask(name="Students1", df=df12, constraints=T12a_CONSTRAINTS, query=T12a_ORIGINAL_SCRIPT_SQL,
                                max_df_length=1000, refineable_predicates=T12a_refineable_predicates)
# Task 12b #

T12b_ORIGINAL_SCRIPT_SQL = """
SELECT * FROM df WHERE "math_score" >= 60.0 AND "parental_level_of_education" IN ('high_school', 'associates_degree');"""

c12b1 = DiversityCardinalityConstraint(attribute='gender', identifier='female', symbol='>=', desired_value=200)
c12b2 = DiversityCardinalityConstraint(attribute='ethnicity', identifier='group_B', symbol='>=', desired_value=100)
c12b3 = DiversityCardinalityConstraint(attribute='all', identifier='yes', symbol='<=', desired_value=600)
T12b_CONSTRAINTS = [c12b1, c12b2, c12b3]


T12b_refineable_attributes = [NumericalAttribute(name="math_score", min_value=0.0, max_value=100.0, step=5.0),
                                CategoricalAttribute(name="parental_level_of_education", categories=df12['parental_level_of_education'].unique())]

T12b_refineable_predicates = [NumericalPredicate(T12b_refineable_attributes[0], None, ">=", 60.0),
                                CategoricalPredicate(T12b_refineable_attributes[1], None, ["high_school", "associates_degree"])]

T12b = DiversityConstraintsTask(name="Students2", df=df12, constraints=T12b_CONSTRAINTS, query=T12b_ORIGINAL_SCRIPT_SQL,
                                max_df_length=1000, refineable_predicates=T12b_refineable_predicates)


# Complex tasks with multiple constraints

## Students Dataset ##

df13 = pd.read_csv('datasets/law_students.csv')
T13_REFINEMENT_OBJECTIVE_STR = """
Sum of absolute normalized distance between (refined value of 'LSAT' and original value of 'LSAT') + (refined value of 'UGPA' and original value of 'UGPA')
"""

T13a_ORIGINAL_SCRIPT_SQL = """
SELECT region, AVG(UGPA) as avg_gpa, AVG(LSAT) as avg_sat, COUNT(*) as size FROM df
WHERE LSAT > 40 AND UGPA > 3.5
GROUP BY region
"""

T13_C1 = AgnosticConstraint(
    query=lambda df: min(df['size']) if len(df) > 0 else 0,
    query_str="(min(df['size']))",
    description="Size of smallest region",
    desired_value=500,
    symbol=">=",
)

T13_C2 = AgnosticConstraint(
    query=lambda df: df['size'].std() / df['size'].mean(),
    query_str="(df['size'].std() / df['size'].mean())",
    description="The CV between the size of each region",
    desired_value=0.4,
    symbol="<="
)

T13_C3 = AgnosticConstraint(
    query=lambda df: df['avg_gpa'].std(),
    query_str="(df['avg_gpa'].std())",
    description="The standard deviation of the average UGPA values across regions",
    desired_value=0.05,
    symbol="<="
)

T13a_CONSTRAINTS = [T13_C1, T13_C2, T13_C3]

T13a_REFINEMENT_OBJECTIVE = get_script_diff_func_sql(T13a_ORIGINAL_SCRIPT_SQL, df13)

T13a_refineable_attributes = [NumericalAttribute(name="LSAT", min_value=11.0, max_value=48.0, step=1),
    NumericalAttribute(name="UGPA", min_value=0.0, max_value=4.2, step=0.1)]

T13a_refineable_predicates = [NumericalPredicate(T13a_refineable_attributes[0], None, ">", 40),
                              NumericalPredicate(T13a_refineable_attributes[1], None, ">", 3.5)]

T13a = AgnosticTask(name="LawStudents1",
                    dataset=df13,
                    query=T13a_ORIGINAL_SCRIPT_SQL,
                    constraints=T13a_CONSTRAINTS,
                    refinement_objective_str=T13_REFINEMENT_OBJECTIVE_STR,
                    refinement_objective=T13a_REFINEMENT_OBJECTIVE,
                    evaluate_constraints_deviation=lambda x: 1,
                    refineable_predicates=T13a_refineable_predicates)


T13b_REFINEMENT_OBJECTIVE_STR = """
Sum of absolute normalized distance between 
(refined value of 'LSAT' and original value of 'LSAT') 
+ (refined value of 'ZFYA' and original value of 'ZFYA')
"""

T13b_ORIGINAL_SCRIPT_SQL = """
SELECT region, gender, AVG(UGPA) AS avg_gpa, AVG(sander_index) AS avg_sander, COUNT(*) AS size FROM df 
WHERE LSAT > 39 AND ZFYA > 0.5
GROUP BY region, gender
"""

T13b_C1 = AgnosticConstraint(
    query=lambda df: min(df['size']) if len(df) > 0 else 0,
    query_str="min(df['size'])",
    description="Size of smallest (region, gender) bucket",
    desired_value=150,
    symbol=">=",
)

T13b_C2 = AgnosticConstraint(
    query=lambda df: df['avg_gpa'].mean(),
    query_str="df['avg_gpa'].mean()",
    description="Standard deviation of avg_gpa across buckets",
    desired_value=3.5,
    symbol=">=",
)

T13b_CONSTRAINTS = [T13b_C1, T13b_C2]

T13b_REFINEMENT_OBJECTIVE = get_script_diff_func_sql(T13b_ORIGINAL_SCRIPT_SQL, df13)

T13b_refineable_attributes = [
    NumericalAttribute(name="LSAT", min_value=11.0, max_value=48.0, step=1),
    NumericalAttribute(name="ZFYA", min_value=-2.0, max_value=2.0, step=0.1)
]

T13b_refineable_predicates = [
    NumericalPredicate(T13b_refineable_attributes[0], None, ">", 39),
    NumericalPredicate(T13b_refineable_attributes[1], None, ">", 0.5)
]

T13b = AgnosticTask(
    name="LawStudents2",
    dataset=df13,
    query=T13b_ORIGINAL_SCRIPT_SQL,
    constraints=T13b_CONSTRAINTS,
    refinement_objective_str=T13b_REFINEMENT_OBJECTIVE_STR,
    refinement_objective=T13b_REFINEMENT_OBJECTIVE,
    evaluate_constraints_deviation=lambda x: 1,
    refineable_predicates=T13b_refineable_predicates
)


df14 = pd.read_csv("datasets/range_query_refinement/texas_2024.csv")

T14a_REFINEMENT_OBJECTIVE_STR = """
Sum of absolute normalized distance between 
(refined value of 'RATE' and original value of 'RATE') 
+ (refined value of 'HRSWKD' and original value of 'HRSWKD')
"""

T14a_ORIGINAL_SCRIPT_SQL = """
SELECT AGENCY, GENDER, AVG(ANNUAL) AS avg_annual, SUM(HRSWKD) AS total_hours FROM df
WHERE MONTHLY > 7000 AND HRSWKD > 25
GROUP BY AGENCY, GENDER
"""

T14a_C1 = AgnosticConstraint(
    query=lambda df: df['avg_annual'].max() / df['avg_annual'].min() if len(df) > 0 else 0,
    query_str="df['avg_annual'].max() / df['avg_annual'].min()",
    description="Ratio of highest to lowest avg_annual across (agency, gender) buckets",
    desired_value=2.0,
    symbol="<=",
)


T14a_C2 = AgnosticConstraint(
    query=lambda df: df['total_hours'].std() / df['total_hours'].mean() if len(df) > 0 else 0,
    query_str="df['total_hours'].std() / df['total_hours'].mean()",
    description="Coefficient of variation of total_hours across buckets",
    desired_value=2.5,
    symbol="<=",
)

T14a_CONSTRAINTS = [T14a_C1, T14a_C2]

T14a_REFINEMENT_OBJECTIVE = get_script_diff_func_sql(T14a_ORIGINAL_SCRIPT_SQL, df14)

T14a_refineable_attributes = [
    NumericalAttribute(name="MONTHLY", min_value=1000, max_value=60000.0, step=1000),
    NumericalAttribute(name="HRSWKD", min_value=0.0, max_value=168.0, step=1.0)
]

T14a_refineable_predicates = [
    NumericalPredicate(T14a_refineable_attributes[0], None, ">", 7000),
    NumericalPredicate(T14a_refineable_attributes[1], None, ">", 25)
]

T14a = AgnosticTask(
    name="TexasTribune1",
    dataset=df14,
    query=T14a_ORIGINAL_SCRIPT_SQL,
    constraints=T14a_CONSTRAINTS,
    refinement_objective_str=T14a_REFINEMENT_OBJECTIVE_STR,
    refinement_objective=T14a_REFINEMENT_OBJECTIVE,
    evaluate_constraints_deviation=lambda x: 1,
    refineable_predicates=T14a_refineable_predicates
)


T14b_REFINEMENT_OBJECTIVE_STR = """
Sum of absolute normalized distance between 
(refined value of 'HRSWKD' and original value of 'HRSWKD') 
+ (refined value of 'ANNUAL' and original value of 'ANNUAL')
"""

T14b_ORIGINAL_SCRIPT_SQL = """
SELECT ETHNICITY,
       AVG(ANNUAL)       AS avg_annual,
       SUM(MONTHLY)      AS total_monthly,
       COUNT(*)          AS size
FROM df
WHERE HRSWKD > 40 AND ANNUAL > 50000
GROUP BY ETHNICITY
"""

T14b_C1 = AgnosticConstraint(
    query=lambda df: df['total_monthly'].min(),
    query_str="df['total_monthly'].min()",
    description="Lowest total monthly value",
    desired_value=3500000,
    symbol=">=",
)

T14b_C2 = AgnosticConstraint(
    query=lambda df: df['size'].std(),
    query_str="df['size'].std()",
    description="Standard deviation of group sizes across ethnicities",
    desired_value=15000,
    symbol="<=",
)

T14b_CONSTRAINTS = [T14b_C1, T14b_C2]

T14b_REFINEMENT_OBJECTIVE = get_script_diff_func_sql(T14b_ORIGINAL_SCRIPT_SQL, df14)

T14b_refineable_attributes = [
    NumericalAttribute(name="HRSWKD", min_value=0.0,    max_value=40.0, step=1.0),
    NumericalAttribute(name="ANNUAL",  min_value=0.0, max_value=200000.0, step=5000.0)
]

T14b_refineable_predicates = [
    NumericalPredicate(T14b_refineable_attributes[0], None, ">", 40),
    NumericalPredicate(T14b_refineable_attributes[1], None, ">", 50000)
]

T14b = AgnosticTask(
    name="TexasTribune2",
    dataset=df14,
    query=T14b_ORIGINAL_SCRIPT_SQL,
    constraints=T14b_CONSTRAINTS,
    refinement_objective_str=T14b_REFINEMENT_OBJECTIVE_STR,
    refinement_objective=T14b_REFINEMENT_OBJECTIVE,
    evaluate_constraints_deviation=lambda x: 1,
    refineable_predicates=T14b_refineable_predicates
)

orders_df = pd.read_csv("datasets/tcph_files/orders.csv")
customer_df = pd.read_csv("datasets/tcph_files/customer.csv")

df15 = {
    "orders": orders_df,
    "customer": customer_df}

T15a_REFINEMENT_OBJECTIVE_STR = """
Sum of absolute normalized distance between 
(refined value of COUNT(*) and original value of COUNT(*)) 
+ (refined lower bound of AVG(o.o_totalprice) and original lower bound of AVG(o.o_totalprice))
+ (refined upper bound of AVG(o.o_totalprice) and original upper bound of AVG(o.o_totalprice))
"""

T15a_ORIGINAL_SCRIPT_SQL = """
SELECT
  c.CUSTKEY AS segment,
  COUNT(*)                 AS num_orders,
  AVG(o.TOTALPRICE)      AS avg_order_total
FROM customer c
JOIN orders   o ON o.CUSTKEY = c.CUSTKEY
GROUP BY c.MKTSEGMENT, c.CUSTKEY
HAVING COUNT(*) >= 6 
   AND AVG(o.TOTALPRICE) >= 30000 AND AVG(o.TOTALPRICE) <= 80000 
"""

A1_C1 = AgnosticConstraint(
    query=lambda df: len(df),
    query_str="len(df)",
    description="Number of customer segments returned",
    desired_value=20,
    symbol=">=",
)

A1_C2 = AgnosticConstraint(
    query=lambda df: float(df["avg_order_total"].max()) if len(df) else 0.0,
    query_str="df['avg_order_total'].max()",
    description="Maximum segment’s average order total",
    desired_value=80000.0,
    symbol=">=",
)

A1_C3 = AgnosticConstraint(
    query=lambda df: (
        float(df["avg_order_total"].max()) / max(float(df["avg_order_total"].min()), 1e-12)
    ) if len(df) else math.inf,
    query_str="df['avg_order_total'].max() / (df['avg_order_total'].min() + 1e-12)",
    description="Stability: max/min of avg_order_total",
    desired_value=2.0,
    symbol="<=",
)

T15a_CONSTRAINTS = [A1_C1, A1_C2, A1_C3]

T15a_refineable_attributes = [
    NumericalAttribute(name="COUNT(*)", min_value=0, max_value=7, step=1),
    NumericalAttribute(name="AVG(o.TOTALPRICE)", min_value=0, max_value=200000, step=10000)
]

T15a_refineable_predicates = [
    NumericalPredicate(T15a_refineable_attributes[0], None, ">=", 6),
    NumericalPredicate(T15a_refineable_attributes[1], None, ">=", 30000),
    NumericalPredicate(T15a_refineable_attributes[1], None, "<=", 80000)
]


T15a_REFINEMENT_OBJECTIVE = get_having_predicate_distance_function(T14b_ORIGINAL_SCRIPT_SQL, T15a_refineable_predicates)


T15a = AgnosticTask(
    name="TPCH1",
    dataset=df15,
    query=T15a_ORIGINAL_SCRIPT_SQL,
    constraints=T15a_CONSTRAINTS,
    refinement_objective_str=T15a_REFINEMENT_OBJECTIVE_STR,
    refinement_objective=T15a_REFINEMENT_OBJECTIVE,
    evaluate_constraints_deviation=lambda x: 1,
    refineable_predicates=T15a_refineable_predicates,
)


lineitem_df = pd.read_csv("datasets/tcph_files/lineitem.csv.gz", compression="gzip")

# CASE A: ISO-like strings or mixed — coerce robustly
lineitem_df["RECEIPTDATE"] = pd.to_datetime(lineitem_df["RECEIPTDATE"], errors="coerce").dt.date
lineitem_df["COMMITDATE"]  = pd.to_datetime(lineitem_df["COMMITDATE"], errors="coerce").dt.date


df15b = {
    "lineitem": lineitem_df,
    "supplier": pd.read_csv("datasets/tcph_files/supplier.csv")
}


T15b_ORIGINAL_SCRIPT_SQL = """
SELECT
  s.NAME                                   AS supplier,
  COUNT(*)                                    AS num_items
FROM supplier s
JOIN lineitem l ON l.SUPPKEY = s.SUPPKEY
GROUP BY s.NAME, s.SUPPKEY
HAVING COUNT(*) >= 45 AND AVG(l.DISCOUNT) <= 0.05;
"""

T15b_REFINEMENT_OBJECTIVE_STR = """
Sum of absolute normalized distance between 
(refined value of COUNT(*) and original value of COUNT(*)) 
+ (refined value of AVG(l.DISCOUNT) and original value of AVG(l.DISCOUNT))
"""

A2_C1 = AgnosticConstraint(
    query=lambda df: len(df),
    query_str="len(df)",
    description="Number of suppliers returned",
    desired_value=1000,
    symbol=">=",
)

A2_C2 = AgnosticConstraint(
    query=lambda df: float(df["num_items"].median()) if len(df) else 0.0,
    query_str="df['num_items'].median()",
    description="Median items per supplier",
    desired_value=100,
    symbol=">=",
)


T15b_CONSTRAINTS = [A2_C1, A2_C2]


T15b_refineable_attributes = [
    NumericalAttribute(name="COUNT(*)", min_value=0, max_value=50, step=5),
    NumericalAttribute(name="AVG(l.DISCOUNT)", min_value=0.01, max_value=0.21, step=0.02)
]

T15b_refineable_predicates = [
    NumericalPredicate(T15b_refineable_attributes[0], None, ">=", 45),
    NumericalPredicate(T15b_refineable_attributes[1], None, "<=", 0.05)
]

T15b_REFINEMENT_OBJECTIVE = get_script_diff_func_sql(T15b_ORIGINAL_SCRIPT_SQL, T15b_refineable_predicates)

T15b = AgnosticTask(
    name="TPCH2",
    dataset=df15b,
    query=T15b_ORIGINAL_SCRIPT_SQL,
    constraints=T15b_CONSTRAINTS,
    refinement_objective_str=T15b_REFINEMENT_OBJECTIVE_STR,
    refinement_objective=T15b_REFINEMENT_OBJECTIVE,
    evaluate_constraints_deviation=lambda x: 1,
    refineable_predicates=T15b_refineable_predicates,
)



supplier_df = pd.read_csv("datasets/tcph_files/supplier.csv")

df16 = {
    "supplier": supplier_df,
    "lineitem": lineitem_df}


T16a_ORIGINAL_SCRIPT_SQL = """
SELECT
  s.NAME                    AS supplier,
  l.SHIPMODE                AS shipmode,
  COUNT(*)                    AS num_lines,
  AVG(l.DISCOUNT)           AS avg_discount
FROM supplier s
JOIN lineitem l ON l.SUPPKEY = s.SUPPKEY
WHERE l.QUANTITY >= 5
  AND l.SHIPMODE IN ('AIR','RAIL')
GROUP BY s.NAME, l.SHIPMODE
HAVING COUNT(*) >= 25
   AND AVG(l.DISCOUNT) <= 0.08;
"""


T16a_REFINEMENT_OBJECTIVE_STR = """
Sum of absolute normalized distance between 
(refined value of l.QUANTITY and original value of l.QUANTITY)
+ (Jaccard distance between refined set of l.SHIPMODE and original set of l.SHIPMODE)
+ (refined value of COUNT(*) and original value of COUNT(*))
+ (refined value of AVG(l.DISCOUNT) and original value of AVG(l.DISCOUNT))
"""

# At least 20 (supplier, shipmode) groups returned
Q1_C1 = AgnosticConstraint(
    query=lambda df: len(df),
    query_str="len(df)",
    description="At least _ supplier–shipmode groups returned",
    desired_value=20,
    symbol=">=",
)

# The highest average discount across groups is at most 10%
Q1_C2 = AgnosticConstraint(
    query=lambda df: float(df["avg_discount"].max()) if len(df) else 0.0,
    query_str="df['avg_discount'].max()",
    description="Max average discount across groups",
    desired_value=0.10,
    symbol="<=",
)

T16a_CONSTRAINTS = [Q1_C1, Q1_C2]

T16a_refineable_attributes = [
    NumericalAttribute(name="l.QUANTITY", min_value=0, max_value=10, step=1),
    CategoricalAttribute(name="l.SHIPMODE", categories=['TRUCK', 'MAIL', 'REG AIR', 'AIR', 'FOB', 'RAIL', 'SHIP']),
    NumericalAttribute(name="COUNT(*)", min_value=0, max_value=100, step=5),
    NumericalAttribute(name="AVG(l.DISCOUNT)", min_value=0.0, max_value=0.2, step=0.01)
]

T16a_refineable_predicates = [
    NumericalPredicate(T16a_refineable_attributes[0], None, ">=", 5),
    CategoricalPredicate(T16a_refineable_attributes[1], None, ['AIR', 'RAIL']),
    NumericalPredicate(T16a_refineable_attributes[2], None, ">=", 25),
    NumericalPredicate(T16a_refineable_attributes[3], None, "<=", 0.08)
]


T16a_REFINEMENT_OBJECTIVE = get_having_predicate_distance_function(T16a_ORIGINAL_SCRIPT_SQL, T16a_refineable_predicates)


T16a = AgnosticTask(
    name="TPCH3",
    dataset=df16,
    query=T16a_ORIGINAL_SCRIPT_SQL,
    constraints=T16a_CONSTRAINTS,
    refinement_objective_str=T16a_REFINEMENT_OBJECTIVE_STR,
    refinement_objective=T16a_REFINEMENT_OBJECTIVE,
    evaluate_constraints_deviation=lambda x: 1,
    refineable_predicates=T16a_refineable_predicates,
)


df16b = {
    "customer": customer_df,
    "orders": orders_df
}


T16b_ORIGINAL_SCRIPT_SQL = """
SELECT
  c.MKTSEGMENT              AS segment,
  o.ORDERPRIORITY           AS priority,
  COUNT(*)                    AS num_orders,
  AVG(o.TOTALPRICE)         AS avg_totalprice
FROM customer c
JOIN orders   o ON o.CUSTKEY = c.CUSTKEY
WHERE o.ORDERSTATUS IN ('F','O')
  AND c.ACCTBAL >= 5
GROUP BY c.MKTSEGMENT, o.ORDERPRIORITY
HAVING COUNT(*) >= 500
   AND AVG(o.TOTALPRICE) >= 150000;
"""

T16b_REFINEMENT_OBJECTIVE_STR = """
Sum of absolute normalized distance between 
(Jaccard distance between refined set of o.ORDERSTATUS and original set of o.ORDERSTATUS)
+ (refined value of c.ACCTBAL and original value of c.ACCTBAL)
+ (refined value of COUNT(*) and original value of COUNT(*))
+ (refined value of AVG(o.TOTALPRICE) and original value of AVG(o.TOTALPRICE))
"""
# At least 10 (segment, priority) groups returned
Q2_C1 = AgnosticConstraint(
    query=lambda df: len(df),
    query_str="len(df)",
    description="Number of segment–priority groups returned",
    desired_value=30,
    symbol=">=",
)

# Weighted average total price (weights=num_orders) is at least 2500
Q2_C2 = AgnosticConstraint(
    query=lambda df: float(df["num_orders"].sum()) if len(df) else 0.0,
    query_str="df['num_orders'].sum()",
    description="Total number of orders across all (segment, priority) groups",
    desired_value=10000,
    symbol=">=",
)
T16b_CONSTRAINTS = [Q2_C1, Q2_C2]

T16b_refineable_attributes = [
    CategoricalAttribute(name="o.ORDERSTATUS", categories=['F', 'O']),
    NumericalAttribute(name="c.ACCTBAL", min_value=0, max_value=10, step=1),
    NumericalAttribute(name="COUNT(*)", min_value=0, max_value=1000, step=100),
    NumericalAttribute(name="AVG(o.TOTALPRICE)", min_value=100000.0, max_value=200000.0, step=10000.0)
]

T16b_refineable_predicates = [
    CategoricalPredicate(T16b_refineable_attributes[0], None, ['F', 'O']),
    NumericalPredicate(T16b_refineable_attributes[1], None, ">=", 5),
    NumericalPredicate(T16b_refineable_attributes[2], None, ">=", 500),
    NumericalPredicate(T16b_refineable_attributes[3], None, ">=", 150000.0)
]

T16b_REFINEMENT_OBJECTIVE = get_having_predicate_distance_function(T16b_ORIGINAL_SCRIPT_SQL, T16b_refineable_predicates)

T16b = AgnosticTask(
    name="TPCH4",
    dataset=df16b,
    query=T16b_ORIGINAL_SCRIPT_SQL,
    constraints=T16b_CONSTRAINTS,
    refinement_objective_str=T16b_REFINEMENT_OBJECTIVE_STR,
    refinement_objective=T16b_REFINEMENT_OBJECTIVE,
    evaluate_constraints_deviation=lambda x: 1,
    refineable_predicates=T16b_refineable_predicates,
)

if __name__ == '__main__':
    top_k_bench = [T1a, T1b, T2a, T2b, T3a, T3b, T4a, T4b]
    range_bench = [T5a, T5b, T6a, T6b, T7a, T7b, T8a, T8b]
    diversity_bench = [T9a, T9b, T10a, T10b, T11a, T11b, T12a, T12b]
    complex_bench = [T13a, T13b, T14a, T14b, T15a, T15b, T16a, T16b]
    RUN_NAME = "oct_18_random_at_1"

    bench_params = [
        # # Main: Actor + Critic
        {"run_name": f"{RUN_NAME}_top_k", "assignment_lm_only": False, "subspace_lm_only_random": False, "epsilon": 0.4, "tasks": top_k_bench},
        {"run_name": f"{RUN_NAME}_range", "assignment_lm_only": True, "subspace_lm_only_random": True, "epsilon": 0.05, "tasks": range_bench},
        {"run_name": f"{RUN_NAME}_diversity", "assignment_lm_only": True, "subspace_lm_only_random": True, "epsilon": 0.0,
         "tasks": diversity_bench},
        {"run_name": f"{RUN_NAME}_complex", "assignment_lm_only": True, "subspace_lm_only_random": True, "epsilon": 0.2,
         "tasks": complex_bench},

        {"run_name": f"{RUN_NAME}_top_k", "assignment_lm_only": False, "subspace_lm_only_random": False, "epsilon": 0.4, "tasks": top_k_bench},
        {"run_name": f"{RUN_NAME}_range", "assignment_lm_only": False, "subspace_lm_only_random": False, "epsilon": 0.05, "tasks": range_bench},
        {"run_name": f"{RUN_NAME}_diversity", "assignment_lm_only": False, "subspace_lm_only_random": False, "epsilon": 0.0,
         "tasks": diversity_bench},
        {"run_name": f"{RUN_NAME}_complex", "assignment_lm_only": False, "subspace_lm_only_random": False, "epsilon": 0.2,
         "tasks": complex_bench},

        # Ablation: Critic Only (Random)
        {"run_name": f"{RUN_NAME}_top_k_subspace_lm_only_random", "assignment_lm_only": False, "subspace_lm_only_random": True,
         "epsilon": 0.4, "tasks": top_k_bench},
        {"run_name": f"{RUN_NAME}_range_subspace_lm_only_random", "assignment_lm_only": False, "subspace_lm_only_random": True, "epsilon": 0.05, "tasks": range_bench},
        {"run_name": f"{RUN_NAME}_diversity_subspace_lm_only_random", "assignment_lm_only": False, "subspace_lm_only_random": True,
         "epsilon": 0.0, "tasks": diversity_bench},
        {"run_name": f"{RUN_NAME}_complex_subspace_lm_only_random", "assignment_lm_only": False, "subspace_lm_only_random": True,
         "epsilon": 0.2, "tasks": complex_bench},

        # Ablation: Actor Only (no Critic)
        {"run_name": f"{RUN_NAME}_top_k_assignment_lm_only", "assignment_lm_only": True, "subspace_lm_only_random": False, "epsilon": 0.4,
         "tasks": top_k_bench},
        {"run_name": f"{RUN_NAME}_range_assignment_lm_only", "assignment_lm_only": True, "subspace_lm_only_random": False, "epsilon": 0.05, "tasks": range_bench},
        {"run_name": f"{RUN_NAME}_diversity_assignment_lm_only", "assignment_lm_only": True, "subspace_lm_only_random": False, "epsilon": 0.0, "tasks": diversity_bench},
        {"run_name": f"{RUN_NAME}_complex_assignment_lm_only", "assignment_lm_only": True, "subspace_lm_only_random": False, "epsilon": 0.2, "tasks": complex_bench},
    ]

    for bench_param in bench_params:
        tasks = bench_param["tasks"]
        print("Starting new benchmark run: ", bench_param["run_name"])
        ACTOR_ONLY = bench_param.get("assignment_lm_only", False)
        CRITIC_ONLY_RANDOM = bench_param.get("subspace_lm_only_random", False)
        ONE_SHOT_MODE = bench_param.get("one_shot", False)
        OUT_PATH = f"exports/gpt41mini/{bench_param['run_name']}.csv"
        LOG_DIR = f"logs/log_dir_{bench_param['run_name']}"
        epsilon = bench_param["epsilon"]
        pandas_dict = {'task_name': [t.name for t in tasks],
                        'original_query': [t.original_query.strip() for t in tasks],
                        'refined_query': [],
                        'refinement_distance': [],
                        'num_successful_refinements': [],
                        'token_use': []}
        is_having = CRITIC_ONLY_RANDOM and "complex" in bench_param["run_name"]

        for i, task in enumerate(tasks):
            best_refined_query = None
            best_distance = float('inf')
            total_token_use = 0
            num_successful_refinements = 0
            num_iterations = 1
            for j in range(num_iterations):
                print(f"Running task {task.name}, iteration {j+1}/{num_iterations}, epsilon {epsilon}")
                try:
                    refined_query, refinement_distance, overall_token_use = run_task(task, epsilon,
                                                                                     perform_analysis=False,
                                                                                     assignment_lm_only_mode=ACTOR_ONLY,
                                                                                     subspace_lm_only_random=CRITIC_ONLY_RANDOM,
                                                                                     one_shot_mode=ONE_SHOT_MODE,
                                                                                     is_having=is_having,
                                                                                     log_dir=LOG_DIR,
                                                                                     random_seed=42,
                                                                                     max_assignments_per_subspace=5,
                                                                                     max_subspace_iters=5)
                except Exception as e:
                    print(f"Error running task {task.name} on iteration {j + 1}: {e}")
                    refined_query = None
                    refinement_distance = float('inf')
                    overall_token_use = 0

                total_token_use += overall_token_use
                if refinement_distance < float('inf'):
                    num_successful_refinements += 1
                if refinement_distance < best_distance:
                    best_distance = refinement_distance
                    best_refined_query = refined_query
            avg_token_use = total_token_use / num_iterations
            pandas_dict['refined_query'].append(best_refined_query)  # top @ 5 refinements with lowest distance
            pandas_dict['refinement_distance'].append(best_distance)  # top @ 5 refinement distances
            pandas_dict['num_successful_refinements'].append(num_successful_refinements)  # number of successful refinements @ 5
            pandas_dict['token_use'].append(avg_token_use)  # average token-use @ 5
        print(pandas_dict)
        df = pd.DataFrame(pandas_dict)
        df.to_csv(OUT_PATH, index=False)
        print(df.head().to_string(index=False))