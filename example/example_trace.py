import os
from agent_trajectory_evaluation import LLMConfig
from agent_trajectory_evaluation.unified import UnifiedEvaluator

print("\n--- Original TRACE Example ---")
config = LLMConfig(provider="openai", model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
evaluator = UnifiedEvaluator(config)

trajectory = [
    {"thought": "Search hotels in Paris", "action": "HotelAPI.search", "action_input": "Paris", "observation": "5 hotels found"},
    {"thought": "Select cheapest", "action": "HotelAPI.select", "action_input": "hotel_id=3", "observation": "Booking confirmed"}
]

results = evaluator.evaluate(trajectory, valid_tools=["HotelAPI.search", "HotelAPI.select"])
print("TRACE Evaluation Metrics:")
for k, v in results.items():
    print(f"  {k}: {v:.3f}")
