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

# Define custom weights with higher values for action, final_answer, action_input, and observation
custom_weights = {'action': 1.0,'action_input': 0.0,'observation': 0.0,'final_answer': 0.0,'thought': 0.0}

# Evaluate with default weights
print("--- Evaluation with Default Weights ---")
scores_default = evaluator.evaluate(pred, gold)
print("GroundTruth KPIs (Default Weights):", scores_default)

# Evaluate with custom weights
print("\n--- Evaluation with Custom Weights ---")
scores_custom = evaluator.evaluate(pred, gold, weights=custom_weights)
print("GroundTruth KPIs (Custom Weights):", scores_custom)
