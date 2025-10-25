import os
from agent_trajectory_evaluation import LLMConfig
from agent_trajectory_evaluation.unified import UnifiedEvaluator

# --- Original TRACE Example ---
# This example demonstrates how to use the UnifiedEvaluator to calculate TRACE metrics
# for a simulated agent trajectory.

print("\n--- Original TRACE Example ---")

# 1. Configure the LLM evaluator
# Replace 'YOUR_API_KEY' with your actual API key for the chosen provider.
# Ensure the corresponding environment variable (e.g., OPENAI_API_KEY) is set,
# or pass the API key directly if the provider supports it.
config = LLMConfig(provider="openai", model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
evaluator = UnifiedEvaluator(config)

# 2. Define a sample agent trajectory
# This trajectory is a list of dictionaries, each representing a step
# with 'thought', 'action', 'action_input', and 'observation'.
# The last step might contain 'final_answer'.
trajectory = [
    {"thought": "Search hotels in Paris", "action": "HotelAPI.search", "action_input": "Paris", "observation": "5 hotels found"},
    {"thought": "Select cheapest", "action": "HotelAPI.select", "action_input": "hotel_id=3", "observation": "Booking confirmed"}
]

# 3. Define valid tools for InstructionError evaluation
valid_tools = ["HotelAPI.search", "HotelAPI.select"]

# 4. Evaluate the trajectory using TRACE metrics
results = evaluator.evaluate(trajectory, valid_tools=valid_tools)

print("TRACE Evaluation Metrics:")
for k, v in results.items():
    print(f"  {k}: {v:.3f}")
