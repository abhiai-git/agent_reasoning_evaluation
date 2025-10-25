import os
import json
from agent_trajectory_evaluation import LLMConfig
from agent_trajectory_evaluation.unified import UnifiedEvaluator
from typing import List, Dict, Any

# --- Google ADK Integration Example ---
# This example demonstrates how to simulate a Google ADK agent's trace
# and convert it into the format compatible with this evaluation package.

# For a real ADK integration, you would use the actual ADK framework
# to run your agent and capture its detailed trace.
# This example provides a simplified simulation of an ADK-like interaction
# and a parser for such a trace.

# Define a mock tool function
def adk_multiply(a: int, b: int) -> int:
    """Multiplies two numbers."""
    return a * b

def adk_add(a: int, b: int) -> int:
    """Adds two numbers."""
    return a + b

# Simulate an ADK-like agent interaction trace
# In a real ADK scenario, this would come from the ADK's tracing mechanism
# or by inspecting the agent's intermediate steps.
def simulate_adk_trace(question: str) -> List[Dict[str, Any]]:
    """
    Simulates a simplified ADK agent's interaction trace for a given question.
    This trace will contain 'thought', 'action', 'action_input', 'observation',
    and potentially 'final_answer'.

    Args:
        question: The input question for the simulated agent.

    Returns:
        A list of dictionaries representing the simulated ADK trace.
    """
    trace = []
    final_answer = None

    if "12 times 3 plus 5" in question:
        # Step 1: Multiply
        thought1 = "I need to multiply 12 by 3 first."
        action1 = "adk_multiply"
        input1 = {"a": 12, "b": 3}
        observation1 = adk_multiply(12, 3)
        trace.append({
            "thought": thought1,
            "action": action1,
            "action_input": input1,
            "observation": observation1
        })

        # Step 2: Add
        thought2 = f"Now I need to add 5 to the result {observation1}."
        action2 = "adk_add"
        input2 = {"a": observation1, "b": 5}
        observation2 = adk_add(observation1, 5)
        trace.append({
            "thought": thought2,
            "action": action2,
            "action_input": input2,
            "observation": observation2
        })
        
        final_answer = str(observation2)
        trace.append({"final_answer": final_answer})

    return trace

# 3. Function to parse ADK trace into evaluation package format
# (In this simulated example, the trace is already in a compatible format,
# but a real parser would extract from ADK-specific objects)
def parse_adk_trace_to_steps(adk_trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Parses a simulated ADK trace (which is already close to the target format)
    into the exact list of steps compatible with the evaluation package.

    Args:
        adk_trace: A list of dictionaries representing the simulated ADK trace.

    Returns:
        A list of dictionaries, each representing a step in the evaluation package format.
    """
    # In a real scenario, this function would extract information from ADK's
    # specific trace objects (e.g., FunctionCall, ToolOutput, LLM generations).
    # For this simulation, we assume the 'simulate_adk_trace' function
    # already produces a list of dictionaries that are almost directly usable.
    
    # We just need to ensure the 'final_answer' is at the end as a separate step
    # and other steps have 'thought', 'action', 'action_input', 'observation'.
    
    parsed_steps = []
    final_answer_step = None

    for step in adk_trace:
        if "final_answer" in step:
            final_answer_step = {"final_answer": step["final_answer"]}
        else:
            parsed_steps.append(step)
            
    if final_answer_step:
        parsed_steps.append(final_answer_step)

    return parsed_steps

# 4. Run the simulated ADK agent and capture trace
print("\n--- Google ADK Agent Trace Example ---")

question = "What is 12 times 3 plus 5?"
print(f"Simulating ADK agent for: '{question}'")
simulated_adk_trace = simulate_adk_trace(question)

# 5. Parse the simulated ADK trace
parsed_trajectory = parse_adk_trace_to_steps(simulated_adk_trace)
print("\nParsed Trajectory (from simulated ADK trace):")
print(json.dumps(parsed_trajectory, indent=2))

# 6. Evaluate the parsed trajectory using the evaluation package
# Ensure GOOGLE_API_KEY is set in environment for actual LLM evaluation
if not os.getenv("GOOGLE_API_KEY"):
    print("GOOGLE_API_KEY environment variable not set. Skipping ADK evaluation with LLM.")
else:
    try:
        evaluator_config = LLMConfig(provider="gemini", model="gemini-pro", api_key=os.getenv("GOOGLE_API_KEY"))
        evaluator = UnifiedEvaluator(evaluator_config)
        
        # Define valid tools for InstructionError evaluation
        valid_tools_for_adk = ["adk_multiply", "adk_add"]

        evaluation_results = evaluator.evaluate(parsed_trajectory, valid_tools=valid_tools_for_adk)
        print("\nEvaluation Results (using this package):")
        for k, v in evaluation_results.items():
            print(f"  {k}: {v:.3f}")

    except Exception as e:
        print(f"An error occurred during ADK example evaluation: {e}")
        print("Please ensure your GOOGLE_API_KEY is set and you have access to the Gemini model.")
