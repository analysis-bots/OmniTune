# OmniTune: A Universal LLM Framework for General Query Refinements

OmniTune is a **universal framework for SQL query refinement** based on **LLM‑driven Optimization by Prompting (OPRO)**. Given a database, an input SQL query, and user‑defined notions of refinement distance and constraint deviation, OmniTune searches for refined queries that remain close to the original while approximately satisfying user constraints.

OmniTune implements the **two‑step OPRO scheme** introduced in the paper:

1. **SubspaceLM** proposes promising *refinement subspaces* over the query predicates.
2. **AssignmentLM** samples concrete refinements within each subspace and evaluates them using constraint and distance functions.

During optimization, OmniTune maintains:

* A **distance–constraint skyline**, and
* A **summarized history** over explored subspaces,

which jointly guide efficient search and enable early stopping.

This repository includes:

* The complete OmniTune system (SubspaceLM, AssignmentLM, skyline, summarized history, early stopping).
* Benchmark drivers for the four instance classes evaluated in the paper: **Top‑k**, **Range**, **Diversity**, and **Complex**.
* Experimental drivers for:

  * Main comparison experiments (LLMs, baselines, random sampling)
  * Ablation studies (Two‑Step OPRO; Skyline/History components)
  * Parameter‑sensitivity experiments over horizon (T) and samples (K)

---

## 🚀 Getting Started

You can run OmniTune either using **Docker** (recommended for reproducibility and artifact evaluation) or directly via **Python 3.11+**.

### Prerequisites

* **Docker** (recommended for reproducibility)
* **Python 3.11+** (for local execution)
* API keys for supported LLM providers (OpenAI, Gemini, Mistral), passed via environment variables or a `.env` file

### Environment Setup

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd omnitune2
   ```

2. Create a `.env` file with your API keys:

   ```bash
   OPENAI_API_KEY=sk-...
   GEMINI_API_KEY=...
   MISTRAL_API_KEY=...
   ```

---

## 🐳 Running with Docker

A `Dockerfile` (and optional `docker-compose.yml`) is provided to reproduce the environment used in the paper.

### 1. Build the Docker Image

```bash
docker build -t omnitune .
```

Or, using Compose:

```bash
docker compose build
```

### 2. Run Experiments

`main.py` is the unified CLI entry point for all experiments.

#### **Direct Docker Run Example (Ablation Study)**

```bash
docker run --rm -it \
  --env-file .env \
  -v $(pwd)/exports:/app/exports \
  -v $(pwd)/logs:/app/logs \
  omnitune \
  python main.py \
    --experiment ablation \
    --model chatgpt \
    --benchmarks top_k range \
    --iterations 5
```

#### **Using Docker Compose (Main Comparison Example)**

```bash
docker compose run --rm omnitune \
  python main.py \
    --experiment comparison \
    --benchmarks all \
    --iterations 5 \
    --plot
```

Outputs are saved under `exports/`, logs under `logs/`.

---

## 🐍 Running Locally (Without Docker)

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Export your API keys:

   ```bash
   export OPENAI_API_KEY=...
   export GEMINI_API_KEY=...
   export MISTRAL_API_KEY=...
   ```

3. Run the CLI:

   ```bash
   python main.py --help
   ```

---

## 🧪 Experiment Types

`main.py` reproduces all experiments from the paper.

### 1. **Main Comparison (`comparison`)**

Evaluates OmniTune vs. baselines across all four benchmark classes:

* **Top‑k** (diverse top‑k selection)
* **Range** (fairness‑aware range predicates)
* **Diversity** (diversity‑constrained refinement tasks)
* **Complex** (novel instances combining multiple constraint types)

Metrics include:

* **Success Rate** ((\psi(\theta) \le \varepsilon))
* **Optimality** (distance (\Delta) to best-known assignments)
* **Token Cost**

### 2. **Ablation Studies (`ablation`)**

Reproduces:

* **Two‑step OPRO vs. naïve OPRO**
* **Effect of Skyline & Summary History** components

### 3. **Parameter Experiments (`parameters`)**

Sweeps over:

* Horizon (T): number of subspace iterations
* Number of assignments sampled per subspace (K)

---

## 🧪 Command Examples

### 1. Run an Ablation Study

```bash
python main.py \
  --experiment ablation \
  --model chatgpt \
  --benchmarks top_k range \
  --iterations 5
```

### 2. Run the Main Comparison (with plots)

```bash
python main.py \
  --experiment comparison \
  --benchmarks all \
  --iterations 5 \
  --plot
```

### 3. Parameter Grid Search (T, K)

```bash
python main.py \
  --experiment parameters \
  --subspaces "1,3,5,7,10" \
  --refinements "1,3,5,7,10" \
  --plot
```

---

## 🎛 CLI Arguments

| Argument             | Description                                                        | Default      |
| -------------------- | ------------------------------------------------------------------ | ------------ |
| `--experiment`, `-e` | Experiment type (`ablation`, `comparison`, `parameters`)           | **Required** |
| `--benchmarks`, `-b` | Benchmark groups (`top_k`, `range`, `diversity`, `complex`, `all`) | `all`        |
| `--model`, `-m`      | LLM family (`chatgpt`, `gemini`, `mistral`)                        | `chatgpt`    |
| `--iterations`, `-n` | Horizon (T): number of subspace iterations                         | `5`          |
| `--output-dir`, `-o` | Output directory                                                   | `exports`    |
| `--seed`             | Random seed                                                        | `42`         |
| `--subspaces`        | Values of (T) for parameter sweeps                                 | `1,3,5,7,10` |
| `--refinements`      | Values of (K) per subspace                                         | `1,3,5,7,10` |
| `--plot`, `-p`       | Generate plots                                                     | `False`      |

---

## 📝 Prompt Template Documentation

OmniTune's LLM‑based refinement uses two collaborative agents:

### • **SubspaceLM** – proposes refinement subspaces (\Theta_t)

### • **AssignmentLM** – generates refined predicate assignments (\theta \in \Theta_t)

Both agents receive structured JSON contexts containing:

* The original SQL query
* Refinable predicates
* Schema summaries and attribute intelligence
* Constraint definitions
* Current skyline summary
* History summaries (local and global)

### Common Template Variables

| Variable                                | Meaning                                                       |
| --------------------------------------- | ------------------------------------------------------------- |
| `{original_query_context}`              | Original SQL query (Q)                                        |
| `{refineable_predicates_context_json}`  | JSON describing refinable predicates                          |
| `{output_constraints_context_json}`     | Constraint deviation function (\psi), tolerance (\varepsilon) |
| `{dataset_schema_context_json}`         | Schema + statistics summary                                   |
| `{refinement_objective_description}`    | Description of refinement distance (\Delta)                   |
| `{attribute_intelligence_context_json}` | Data-driven predicate statistics                              |
| `{current_subspace_json}`               | Current subspace (\Theta_t)                                   |
| `{refinement_history_json}`             | Local history in (\Theta_t)                                   |
| `{query_skyline}` / `{skyline}`         | Current skyline summary                                       |
| `{subspace_history_json}`               | History over previously explored subspaces                    |

---

## 🧠 Core Prompt Sketches

OmniTune utilizes a dual-agent architecture: the **AssignmentLM** and the **SubspaceLM**. Below are the core prompt templates used to guide these agents, located in `opro/opro_prompt_templates.py`.

### 1. AssignmentLM

The AssignmentLM is responsible for generating specific predicate assignments within a given subspace $\Theta_t$.

**System Prompt:**
```text
You are the AssignmentLM. You generate candidate predicate assignments within a provided subspace Θ_t.

**PRELIMINARY DEFINITIONS:**
[... Definitions of Predicates, Valid Ranges, Subspaces, Constraints, and Refinement Distance ...]

**Static task context (fixed for this run):**
- Original Query:
{original_query_context}
- Refineable Predicates:
{refineable_predicates_context_json}

- Output Constraints:
{output_constraints_context_json}

- Dataset Schema:
{dataset_schema_context_json}

- Refinement Distance Objective:
{refinement_objective_description}

- Attribute Intelligence (Data-Informed Stats):
{attribute_intelligence_context_json}

**What you will receive in refinement prompts:**
- The current subspace Θ_t (allowed ranges/sets for every predicate).
- Local history for this subspace (recent attempts and outcomes).
- A skyline summary (current (Δ, ψ_ε) trade-offs) to avoid regressions.

**Your job:** Produce assignments consistent with Θ_t that can satisfy all constraints and improve the skyline (reduce distance while keeping constraint deviation near 0). Keep SQL structure unchanged except predicate values and stay within valid ranges.

**Response format (always JSON in a fenced block):**
{
  "reasoning_concise": "brief rationale",
  "selected_refinement": "<complete_sql_query>",
  "selected_predicate_values": {
    "<predicate_id_1>": <value>,
    "<predicate_id_2>": [<set_of_values>]
  }
}
```

**Refinement Suggestion Prompt:**
```text
AssignmentLM: propose a refinement using the current subspace Θ_t. Keep SQL structure unchanged; values must stay within Θ_t.

- Skyline view (current (Δ, ψ_ε) trade-offs):
{query_skyline}
 
- Current subspace Θ_t (allowed ranges/sets):
{current_subspace_json}

- Local history within Θ_t (recent attempts):
{refinement_history_json}

Response format (JSON, fenced):
{
  "reasoning_concise": "why this assignment can satisfy constraints and improve the skyline",
  "selected_refinement": "<complete_sql_query>",
  "selected_predicate_values": {
    "<predicate_id_1>": <value>,
    "<predicate_id_2>": [<set_of_values>]
  }
}

```

### 2. SubspaceLM

The SubspaceLM acts as a manager, proposing new search subspaces likely to contain better solutions based on the history of exploration.

**System Prompt:**
```text
You are the SubspaceLM. You propose predicate subspaces that are likely to contain assignments improving the (Δ, ψ_ε) skyline.

**PRELIMINARY DEFINITIONS:**
[... Definitions of Predicates, Valid Ranges, Subspaces, Constraints, and Refinement Distance ...]

**Static task context (fixed for this run):**
[... Original Query, Predicates, Constraints, Schema, etc. ...]

**What you will receive in selection prompts:**
- Skyline summary S⁺(Q) describing current best trade-offs.
- Subspace history summaries (performance of prior subspaces).

**Your job:** Propose a predicate subspace that includes every refineable predicate,
respects valid ranges, and is likely to yield assignments that satisfy constraints and improve the skyline.

**Output format (JSON in a fenced block):**
{
  "patterns_observed": "<brief signal from prior subspaces>",
  "selected_predicate_subspace": {
    "<predicate_id_1>": [<min_value>, <max_value>],
    "<predicate_id_2>": [<contained_set>, <containing_set>]
  },
  "reasoning_concise": "short strategy statement",
  "breakthrough_potential": <0.0-1.0>,
  "expected_efficiency": "<success_rate>% trials → constraint satisfaction",
  "novelty_score": <0.0-1.0>
}
```

**Validation:** Ensure each numerical range contains ≥5 feasible values (respecting step) and categorical sets obey contained ⊆ values ⊆ containing. Avoid already-exhausted regions and prefer subspaces that can reduce constraint deviation before distance.

**Subspace Suggestion Prompt:**
```text
SubspaceLM: propose a predicate subspace likely to contain assignments that improve the (Δ, ψ_ε) skyline.

- Skyline view (current (Δ, ψ_ε) trade-offs):
 {skyline}

- Historical subspace performance (signals from prior subspaces):
 {subspace_history_json}

Guidance:
- Respect valid ranges for every predicate.
- Ensure each numerical range contains at least five feasible values.
- Prefer regions not yet explored that could satisfy constraints and reduce distance.

Response format (JSON, fenced):
{
  "patterns_observed": "<brief summary of signals from subspace history>",
  "selected_predicate_subspace": {
    "<predicate_id_1>": [<min_value>, <max_value>],
    "<predicate_id_2>": [<contained_set>, <containing_set>]     # contained set should be as small as possible!
  },
  "reasoning_concise": "short strategy statement",
  "breakthrough_potential": <0.0-1.0>,
  "expected_efficiency": "<success_rate>% trials → constraint satisfaction",
  "novelty_score": <0.0-1.0>
}
```
