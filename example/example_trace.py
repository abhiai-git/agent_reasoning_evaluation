from agent_trajectory_evaluation import LLMConfig
from agent_trajectory_evaluation.unified import UnifiedEvaluator

# TRACE (LLM-as-Judge) Evaluation Example — No Ground Truth

config = LLMConfig(provider="openai", model="gpt-4o-mini")
evaluator = UnifiedEvaluator(config)

trajectory = [
    {"thought": "Search for hotels in Paris", "action": "HotelAPI.search", "action_input": "Paris", "observation": "5 hotels found"},
    {"thought": "Select the cheapest hotel", "action": "HotelAPI.select", "action_input": "hotel_id=3", "observation": "Booking confirmed"}
]

# Only TRACE metrics (Efficiency, Hallucination, Adaptivity, InstructionError)
results = evaluator.evaluate(trajectory, valid_tools=["HotelAPI.search", "HotelAPI.select"])

print("TRACE Evaluation Results (No Ground Truth):")
for k, v in results.items():
    print(f"  {k}: {v:.3f}")
