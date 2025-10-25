# Agent Trajectory Evaluation

**A Unified Framework for Evaluating LLM Agent Reasoning, Trajectories, and Telemetry**

---

## Overview

`agent_trajectory_evaluation` provides a **modular and extensible framework** for evaluating the reasoning quality, efficiency, and fidelity of **tool-using LLM agents**.

It supports two major evaluation paradigms:

1. **LLM-as-Judge (TRACE)** — evaluates reasoning quality **without ground truth**, using intrinsic metrics such as Efficiency, Hallucination, and Adaptivity.  
2. **GroundTruth-Based Evaluation (GRACE-style)** — compares agent trajectories or telemetry logs against reference data, using a **weighted overlap score** that incorporates Exact and Fuzzy matching for individual components of each step and the final answer.

The toolkit unifies both under a single, composable API that is **multi-provider compatible** (OpenAI, Gemini, Claude, Bedrock).

---

## Key Features

| Category | Features |
|-----------|-----------|
| **Evaluation Modes** | • TRACE (No GroundTruth)  • GroundTruth Comparison  • Unified Evaluation |
| **Metrics** | Efficiency, Hallucination, Adaptivity, InstructionError, WeightedOverlapScore |
| ☁️ **Multi-Provider Support** | OpenAI (GPT-4/4o), Google Gemini, Anthropic Claude, AWS Bedrock |
| **Parallel Evaluation** | Async batch scoring for large-scale datasets |
| **Embedding-based Similarity** | Uses SentenceTransformers for semantic fuzzy scoring |
| **KPI Aggregation** | Returns structured metrics for dashboards and research |
| **Plug-and-Play Design** | Minimal setup, clean API, extensible for future agents or datasets |

---

## Installation

### Option 1 — From GitHub (Recommended)

```bash
pip install git+https://github.com/abhiai-git/agent_trajectory_evaluation.git
```

### Option 2 — From Local Zip 

```bash
pip install agent_trajectory_evaluation.zip
```

### Option 3 — Development / Editable Mode

```bash
git clone https://github.com/abhiai-git/agent_trajectory_evaluation.git
cd agent_trajectory_evaluation
pip install -e .
```
### Example 1 — TRACE (LLM-as-Judge, No GroundTruth)

Evaluates reasoning traces for Efficiency, Hallucination, Adaptivity, and Instruction Error
using any supported LLM as the evaluator (OpenAI, Gemini, Claude, Bedrock).

```python
from agent_trajectory_evaluation import LLMConfig
from agent_trajectory_evaluation.unified import UnifiedEvaluator

# --- Configuration for LLM Evaluator ---
# Replace 'YOUR_API_KEY' with your actual API key for the chosen provider.
# Ensure the corresponding environment variable (e.g., OPENAI_API_KEY) is set,
# or pass the API key directly if the provider supports it.

# Example for OpenAI:
config = LLMConfig(provider="openai", model="gpt-4o-mini", api_key="YOUR_OPENAI_API_KEY")
# config = LLMConfig(provider="openai", model="gpt-4-turbo", api_key="YOUR_OPENAI_API_KEY")

# Example for Google Gemini:
# config = LLMConfig(provider="gemini", model="gemini-pro", api_key="YOUR_GEMINI_API_KEY")
# config = LLMConfig(provider="gemini", model="gemini-1.5-pro", api_key="YOUR_GEMINI_API_KEY")

# Example for Anthropic Claude:
# config = LLMConfig(provider="claude", model="claude-3-sonnet-20240229", api_key="YOUR_ANTHROPIC_API_KEY")
# config = LLMConfig(provider="claude", model="claude-3-opus-20240229", api_key="YOUR_ANTHROPIC_API_KEY")

# Example for AWS Bedrock (requires AWS credentials configured):
# config = LLMConfig(provider="bedrock", model="anthropic.claude-3-sonnet-20240229-v1:0", region_name="us-east-1")
# config = LLMConfig(provider="bedrock", model="amazon.titan-text-express-v1", region_name="us-east-1")

evaluator = UnifiedEvaluator(config)

trajectory = [
    {"thought": "Search hotels in Paris", "action": "HotelAPI.search", "action_input": "Paris", "observation": "5 hotels found"},
    {"thought": "Select cheapest", "action": "HotelAPI.select", "action_input": "hotel_id=3", "observation": "Booking confirmed"}
]

results = evaluator.evaluate(trajectory, valid_tools=["HotelAPI.search", "HotelAPI.select"])
print("TRACE Evaluation Metrics:")
for k, v in results.items():
    print(f"  {k}: {v:.3f}")
```
#### Output
```yaml
TRACE Evaluation Metrics:
  Efficiency: 0.600
  Hallucination: 1.000
  Adaptivity: 0.000
  InstructionError: 1.000
```

### Example 2 — GroundTruth Comparison

Compares a model-generated trajectory with a gold reference trajectory and final output.

```python
from agent_trajectory_evaluation.groundtruth import GroundTruthEvaluator

pred = {
    "steps": [
        {"action": "WeatherAPI.query", "action_input": "San Francisco", "observation": "Sunny"},
        {"action": "CalendarAPI.add_event", "action_input": "Picnic at Golden Gate Park", "observation": "Event added"}
    ],
    "final_answer": "Picnic scheduled for sunny day in San Francisco"
}

gold = {
    "steps": [
        {"action": "WeatherAPI.query", "action_input": "SF", "observation": "Sunny"},
        {"action": "CalendarAPI.add_event", "action_input": "Picnic @ Golden Gate Park", "observation": "Event added successfully"}
    ],
    "final_answer": "Event scheduled for a sunny day in SF"
}

evaluator = GroundTruthEvaluator()

# Custom weights can be provided (optional)
custom_weights = {
    'action': 0.30,
    'action_input': 0.25,
    'observation': 0.20,
    'final_answer': 0.25,
    'thought': 0.0 # Thoughts are not directly compared for overlap score
}

# Evaluate with default weights
print("--- Evaluation with Default Weights ---")
scores_default = evaluator.evaluate(pred, gold)
print("GroundTruth KPIs (Default Weights):", scores_default)

# Evaluate with custom weights
print("\n--- Evaluation with Custom Weights ---")
scores_custom = evaluator.evaluate(pred, gold, weights=custom_weights)
print("GroundTruth KPIs (Custom Weights):", scores_custom)
```
#### Output (Example with default weights)
```
{
  'OverallWeightedOverlapScore': 0.785
}
```

The `GroundTruthEvaluator` now uses a `multi_step_overlap` function that calculates a single weighted score. This score considers the similarity of `action`, `action_input`, `observation` for each step, and the `final_answer`. Weights for these components are configurable, allowing users to prioritize certain aspects of the trajectory. By default, `action` and `final_answer` are given higher weights.

### Example 3 — Unified Evaluation (TRACE + GroundTruth)

```python
from agent_trajectory_evaluation import LLMConfig
from agent_trajectory_evaluation.unified import UnifiedEvaluator

cfg = LLMConfig(provider="claude", model="claude-3-sonnet-20241022")
eval = UnifiedEvaluator(cfg)

trace = [
    {"thought": "Find cheapest NYC->LAX flight", "action": "FlightAPI.search", "action_input": "NYC-LAX", "observation": "3 flights found"},
    {"thought": "Select flight_id=2", "action": "FlightAPI.select", "action_input": "flight_id=2", "observation": "Booked"}
]

reference = {
    "steps": [
        {"action": "FlightAPI.search", "action_input": "NYC to LAX", "observation": "3 flights found"},
        {"action": "FlightAPI.select", "action_input": "flight_id=2", "observation": "Booking confirmed"}
    ],
    "final_answer": "Flight booked to LAX"
}

metrics = eval.evaluate(trace, valid_tools=["FlightAPI.search", "FlightAPI.select"], reference=reference)
print(metrics)
```

### TRACE Metrics Formulas

The TRACE framework evaluates agent trajectories across four key dimensions:

1.  **Efficiency ($\text{Eff}(\mathcal{T})$)**: Quantifies the proportion of necessary evidence in a successful trajectory.
    $$\text{Eff}(\mathcal{T}) = \frac{ \|\mathcal{E}_{min}\| }{ \|\mathcal{E}_{n}\| } = 1 - \frac{ \|\mathcal{E}_{unnecessary}\| }{ \|\mathcal{E}_{n}\| }$$
    Where $\mathcal{E}_{min}$ is the minimal set of evidence required to deduce the final answer, and $\|\mathcal{E}_{n}\|$ is the total evidence collected.

2.  **Hallucination ($\text{H}(\mathcal{T})$)**: Measures the average rate of thoughts that are not grounded in the accumulated evidence.
    $$\text{H}(\mathcal{T}) = \frac{ \sum_{t=1}^{n} H(s_t) }{ n }$$
    Where $H(s_t) = 1$ if the thought $th_t$ at step $s_t$ is not logically derivable from the evidence bank $\mathcal{E}_{t-1}$, and $0$ otherwise.

3.  **Adaptivity ($\text{Adp}(\mathcal{T})$)**: Assesses the agent's ability to recover from tool failures.
    $$\text{Adp}(s_{t+1}) = 1 \quad \text{if } th_{t+1} \text{ acknowledges failure and } a_{t+1} \text{ is a sensible alternative, else } 0$$
    The overall adaptivity score is the average of $\text{Adp}(s_{t+1})$ for all tool failure events.

4.  **Instruction Error (Inst.)**: Represents the ratio of steps where the agent fails to select an existing tool or uses an incorrect argument format.
    $$Inst. = \frac{ \text{Count of invalid actions or input formats} }{ \text{Total number of steps in trajectory} }$$

### Theoretical Foundation
This framework synthesizes ideas from three major research efforts:

- TRACE: Evaluating Tool-Enabled Agent Reasoning Zhou, J., Chen, H.,Pan, L., et al., 2024 arXiv:2404.06626

- GRACE: Grounded Reasoning Agent Calibration and Evaluation Li, S., Zhao, T., et al., 2024 arXiv:2406.01856

- BIG-Bench Hard / HELM Agent Evaluation Framework Srivastava, A., Leike, J., et al., 2023 arXiv:2306.11644

### Citation
```bibtex
@software{agent_trajectory_evaluation,
  title  = {agent_trajectory_evaluation: Unified LLM Agent Evaluation Toolkit},
  author = {Abhishek Bhardwaj},
  year   = {2025},
  url    = {https://github.com/abhiai-git/agent_trajectory_evaluation},
  note   = {Evaluates reasoning trajectories via TRACE and GroundTruth frameworks}
}
```
