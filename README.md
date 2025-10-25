# Agent Trajectory Evaluation

**A Unified Framework for Evaluating LLM Agent Reasoning, Trajectories, and Telemetry**

---

## Overview

`agent_trajectory_evaluation` provides a **modular and extensible framework** for evaluating the reasoning quality, efficiency, and fidelity of **tool-using LLM agents**.

It supports two major evaluation paradigms:

1. **LLM-as-Judge (TRACE)** — evaluates reasoning quality **without ground truth**, using intrinsic metrics such as Efficiency, Hallucination, and Adaptivity.  
2. **GroundTruth-Based Evaluation (GRACE-style)** — compares agent trajectories or telemetry logs against reference data, using a **weighted overlap score** that incorporates Exact and Fuzzy matching for individual components of each step and the final answer. This evaluation can be performed in either **ordered** or **unordered** mode, allowing flexibility in how step sequences are compared.

The toolkit unifies both under a single, composable API that is **multi-provider compatible** (OpenAI, Gemini, Claude, Bedrock).

---

## Key Features

| Category | Features |
|-----------|-----------|
| **Evaluation Modes** | • TRACE (No GroundTruth)  • GroundTruth Comparison  • Unified Evaluation |
| **Metrics** | Efficiency, Hallucination, Adaptivity, InstructionError, WeightedOverlapScore |
|  **Multi-Provider Support** | OpenAI (GPT-4/4o), Google Gemini, Anthropic Claude, AWS Bedrock |
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

# Evaluate with default weights (ordered matching)
print("--- Evaluation with Default Weights (Ordered Matching) ---")
scores_default_ordered = evaluator.evaluate(pred, gold)
print("GroundTruth KPIs (Default Weights, Ordered):", scores_default_ordered)

# Evaluate with custom weights (ordered matching)
print("\n--- Evaluation with Custom Weights (Ordered Matching) ---")
scores_custom_ordered = evaluator.evaluate(pred, gold, weights=custom_weights)
print("GroundTruth KPIs (Custom Weights, Ordered):", scores_custom_ordered)

# Example for unordered trajectory matching
# Let's create a slightly reordered prediction for demonstration
pred_unordered = {
    "steps": [
        {"action": "CalendarAPI.add_event", "action_input": "Picnic at Golden Gate Park", "observation": "Event added"},
        {"action": "WeatherAPI.query", "action_input": "San Francisco", "observation": "Sunny"}
    ],
    "final_answer": "Picnic scheduled for sunny day in San Francisco"
}

print("\n--- Evaluation with Unordered Trajectory Matching (Default Weights) ---")
scores_unordered = evaluator.evaluate(pred_unordered, gold, trajectory_match_mode="unordered")
print("GroundTruth KPIs (Unordered Mode):", scores_unordered)

print("\n--- Evaluation with Ordered Trajectory Matching (Default Weights, for comparison) ---")
scores_ordered_reordered_pred = evaluator.evaluate(pred_unordered, gold, trajectory_match_mode="ordered")
print("GroundTruth KPIs (Ordered Mode with reordered pred):", scores_ordered_reordered_pred)
```
#### Output (Example with default weights and ordered matching)
```
{
  'OverallWeightedOverlapScore': 0.785
}
```
#### Output (Example with default weights and unordered matching)
```
{
  'OverallWeightedOverlapScore': 0.850
}
```

The `GroundTruthEvaluator` now uses a `multi_step_overlap` function that calculates a single weighted score. This score considers the similarity of `action`, `action_input`, `observation` for each step, and the `final_answer`. Weights for these components are configurable, allowing users to prioritize certain aspects of the trajectory. By default, `action` and `final_answer` are given higher weights. The `trajectory_match_mode` parameter allows choosing between 'ordered' (default) and 'unordered' matching for trajectory steps.

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

### Example 4 — LangChain Integration

This example demonstrates how to capture a LangChain agent's trace and convert it into a format compatible with this evaluation package, then evaluate it using TRACE metrics. The full example code is located in `agent_trajectory_evaluation/example/example_langchain.py`.

#### Output (Example from running `example_langchain.py`)
```
# ... (output from LangChain agent execution) ...

Parsed Trajectory (from LangChain trace):
[
  {
    "thought": "I need to use the multiply tool.",
    "action": "multiply",
    "action_input": {
      "a": 12,
      "b": 3
    },
    "observation": 36
  },
  {
    "thought": "I need to use the add tool.",
    "action": "add",
    "action_input": {
      "a": 36,
      "b": 5
    },
    "observation": 41
  },
  {
    "final_answer": "41"
  }
]

Evaluation Results (using this package):
  Efficiency: 0.600
  Hallucination: 0.000
  Adaptivity: 1.000
  InstructionError: 0.000
```

### Example 5 — Google ADK Integration

This example demonstrates how to simulate a Google ADK agent's trace and convert it into a format compatible with this evaluation package, then evaluate it using TRACE metrics. The full example code is located in `agent_trajectory_evaluation/example/example_adk.py`.

#### Output (Example from running `example_adk.py`)
```
# ... (output from ADK agent execution) ...

Parsed Trajectory (from simulated ADK trace):
[
  {
    "thought": "I need to multiply 12 by 3 first.",
    "action": "adk_multiply",
    "action_input": {
      "a": 12,
      "b": 3
    },
    "observation": 36
  },
  {
    "thought": "Now I need to add 5 to the result 36.",
    "action": "adk_add",
    "action_input": {
      "a": 36,
      "b": 5
    },
    "observation": 41
  },
  {
    "final_answer": "41"
  }
]

Evaluation Results (using this package):
  Efficiency: 0.600
  Hallucination: 0.000
  Adaptivity: 1.000
  InstructionError: 0.000
```

### Example 6 — CrewAI Integration

This example demonstrates how to simulate a CrewAI agent's trace and convert it into a format compatible with this evaluation package, then evaluate it using TRACE metrics. The full example code is located in `agent_trajectory_evaluation/example/example_crewai.py`.

```python
import os
import json
from agent_trajectory_evaluation import LLMConfig
from agent_trajectory_evaluation.unified import UnifiedEvaluator

# ... (CrewAI simulation and parsing functions as in example_crewai.py) ...

# Example usage:
if __name__ == "__main__":
    # Ensure OPENAI_API_KEY is set in environment
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY environment variable not set. Skipping CrewAI evaluation with LLM.")
    else:
        try:
            # This part would typically involve running your CrewAI agent
            # and capturing its trace, similar to the content of example_crewai.py
            # For brevity, we'll show the evaluation part directly.

            # Example parsed trajectory (from a hypothetical CrewAI run)
            parsed_trajectory = [
                {"thought": "I need to find the current year first using the SearchTool.", "action": "SearchTool", "action_input": {"query": "current year"}, "observation": "The current year is 2025."},
                {"thought": "The current year is 2025. I need to add 10 to it using the Calculator.", "action": "Calculator", "action_input": {"operation": "2025 + 10"}, "observation": "2035"},
                {"final_answer": "2035"}
            ]
            print("\nParsed Trajectory (from simulated CrewAI trace):")
            print(json.dumps(parsed_trajectory, indent=2))

            evaluator_config = LLMConfig(provider="openai", model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
            evaluator = UnifiedEvaluator(evaluator_config)
            
            valid_tools_for_crewai = ["Calculator", "SearchTool"]

            evaluation_results = evaluator.evaluate(parsed_trajectory, valid_tools=valid_tools_for_crewai)
            print("\nEvaluation Results (using this package):")
            for k, v in evaluation_results.items():
                print(f"  {k}: {v:.3f}")

        except Exception as e:
            print(f"An error occurred during CrewAI example evaluation: {e}")
            print("Please ensure your OPENAI_API_KEY is set and CrewAI dependencies are correctly installed.")
```
#### Output (Example from running `example_crewai.py`)
```
# ... (output from CrewAI agent execution) ...

Parsed Trajectory (from simulated CrewAI trace):
[
  {
    "thought": "I need to find the current year first using the SearchTool.",
    "action": "SearchTool",
    "action_input": {
      "query": "current year"
    },
    "observation": "The current year is 2025."
  },
  {
    "thought": "The current year is 2025. I need to add 10 to it using the Calculator.",
    "action": "Calculator",
    "action_input": {
      "operation": "2025 + 10"
    },
    "observation": "2035"
  },
  {
    "final_answer": "2035"
  }
]

Evaluation Results (using this package):
  Efficiency: 0.600
  Hallucination: 0.000
  Adaptivity: 1.000
  InstructionError: 0.000
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
This framework synthesizes ideas from several major research efforts:

- TRACE: Evaluating Tool-Enabled Agent Reasoning Kim, W., Park, S., In, Y., et al., 2025 arXiv:2510.02837
- ACPBench: Reasoning about Action, Change, and Planning Kokel, H., Katz, M., Srinivas, K., et al., 2024 arXiv:2410.05669
- NATURALPLAN: Benchmarking LLMs on Natural Language Planning Zheng, H. S., Mishra, S., Zhang, H., et al., 2024 arXiv:2406.04520
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

@article{Kim2025BeyondTF,
  title={BEYOND THE FINAL ANSWER: EVALUATING THE REASONING TRAJECTORIES OF TOOL-AUGMENTED AGENTS},
  author={Wonjoong Kim and Sangwu Park and Yeonjun In and Sein Kim and Dongha Lee and Chanyoung Park},
  journal={arXiv preprint arXiv:2510.02837},
  year={2025}
}

@article{Kokel2024ACPBenchRA,
  title={ACPBench: Reasoning about Action, Change, and Planning},
  author={Harsha Kokel and Michael Katz and Kavitha Srinivas and Shirin Sohrabi},
  journal={arXiv preprint arXiv:2410.05669},
  year={2024}
}

@article{Zheng2024NATURALPLANBL,
  title={NATURALPLAN: Benchmarking LLMs on Natural Language Planning},
  author={Huaixiu Steven Zheng and Swaroop Mishra and Hugh Zhang and Xinyun Chen and Minmin Chen and Azade Nova and Le Hou and Heng-Tze Cheng and Quoc V. Le and Ed H. Chi and Denny Zhou},
  journal={arXiv preprint arXiv:2406.04520},
  year={2024}
}
```
**Full Citations:**
- Kim, W., Park, S., In, Y., Kim, S., Lee, D., & Park, C. (2025). BEYOND THE FINAL ANSWER: EVALUATING THE REASONING TRAJECTORIES OF TOOL-AUGMENTED AGENTS. *arXiv preprint arXiv:2510.02837*.
- Kokel, H., Katz, M., Srinivas, K., & Sohrabi, S. (2024). ACPBench: Reasoning about Action, Change, and Planning. *arXiv preprint arXiv:2410.05669*.
- Zheng, H. S., Mishra, S., Zhang, H., Chen, X., Chen, M., Nova, A., Hou, L., Cheng, H. T., Le, Q. V., Chi, E. H., & Zhou, D. (2024). NATURALPLAN: Benchmarking LLMs on Natural Language Planning. *arXiv preprint arXiv:2406.04520*.
