from agent_trajectory_evaluation.groundtruth import GroundTruthEvaluator
from typing import Dict, List, Any, Optional

pred: Dict[str, Any] = {
    "steps": [
        {"action": "WeatherAPI.query", "action_input": "San Francisco", "observation": "Sunny"},
        {"action": "CalendarAPI.add_event", "action_input": "Picnic at Golden Gate Park", "observation": "Event added"}
    ],
    "final_answer": "Picnic scheduled for sunny day in San Francisco"
}

gold: Dict[str, Any] = {
    "steps": [
        {"action": "WeatherAPI.query", "action_input": "SF", "observation": "Sunny"},
        {"action": "CalendarAPI.add_event", "action_input": "Picnic @ Golden Gate Park", "observation": "Event added successfully"}
    ],
    "final_answer": "Event scheduled for a sunny day in SF"
}

evaluator = GroundTruthEvaluator()

# Define custom weights with higher values for action, final_answer, action_input, and observation
custom_weights: Dict[str, float] = {
    'action': 0.30,
    'action_input': 0.25,
    'observation': 0.20,
    'final_answer': 0.25,
    'thought': 0.0
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
pred_unordered: Dict[str, Any] = {
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
