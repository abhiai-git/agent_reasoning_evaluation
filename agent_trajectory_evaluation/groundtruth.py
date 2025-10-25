import numpy as np
import difflib
from typing import Dict, List, Any, Optional
from sentence_transformers import SentenceTransformer, util
from fuzzywuzzy import fuzz
import json

class GroundTruthEvaluator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)

    @staticmethod
    def exact_match(pred: str, gold: str) -> float:
        return 1.0 if pred.strip().lower() == gold.strip().lower() else 0.0

    def fuzzy_match(self, pred: str, gold: str, method: str = "semantic") -> float:
        if method == "semantic":
            emb_pred = self.embedder.encode(pred, convert_to_tensor=True)
            emb_gold = self.embedder.encode(gold, convert_to_tensor=True)
            return float(util.cos_sim(emb_pred, emb_gold))
        elif method == "token":
            # Using fuzzywuzzy for token-based fuzzy matching
            return fuzz.ratio(pred.lower(), gold.lower()) / 100.0
        else:
            raise ValueError("method must be 'semantic' or 'token'")

    def multi_step_overlap(
        self,
        pred_trajectory: Dict[str, Any],
        gold_trajectory: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculates a weighted overlap score between two trajectories, considering
        exact and fuzzy matches for action, action_input, observation, and final_answer.

        Args:
            pred_trajectory: A dictionary representing the predicted trajectory,
                             including 'steps' (list of dicts) and 'final_answer'.
            gold_trajectory: A dictionary representing the ground truth trajectory.
            weights: A dictionary of weights for different components.
                     Keys: 'action', 'action_input', 'observation', 'final_answer'.
                     Values: float (sum should ideally be 1.0 for normalized scores).
                     Default weights are used if not provided.

        Returns:
            A float representing the weighted overlap score (0.0 to 1.0).
        """
        # Default weights with higher values for action, final_answer, action_input, observation
        default_weights = {
            'action': 0.25,
            'action_input': 0.20,
            'observation': 0.20,
            'final_answer': 0.35,
            'thought': 0.0 # Thoughts are not directly compared for overlap score
        }
        component_weights = {**default_weights, **(weights if weights is not None else {})}
        
        # Normalize weights if they don't sum to 1
        total_weight_sum = sum(component_weights.values())
        if total_weight_sum == 0:
            return 0.0 # Avoid division by zero if all weights are zero
        normalized_weights = {k: v / total_weight_sum for k, v in component_weights.items()}

        pred_steps = pred_trajectory.get("steps", [])
        gold_steps = gold_trajectory.get("steps", [])

        min_len = min(len(pred_steps), len(gold_steps))
        step_scores = []
        
        for i in range(min_len):
            step1 = pred_steps[i]
            step2 = gold_steps[i]
            current_step_score = 0.0
            current_step_component_weight_sum = 0.0

            # Compare 'action'
            action1 = step1.get('action', '')
            action2 = step2.get('action', '')
            if normalized_weights.get('action', 0) > 0:
                exact_match = 1.0 if action1 == action2 else 0.0
                fuzzy_match = fuzz.ratio(str(action1), str(action2)) / 100.0
                current_step_score += normalized_weights['action'] * (exact_match * 0.7 + fuzzy_match * 0.3)
                current_step_component_weight_sum += normalized_weights['action']

            # Compare 'action_input'
            input1 = step1.get('action_input', {})
            input2 = step2.get('action_input', {})
            if normalized_weights.get('action_input', 0) > 0:
                str_input1 = json.dumps(input1, sort_keys=True)
                str_input2 = json.dumps(input2, sort_keys=True)
                exact_match = 1.0 if str_input1 == str_input2 else 0.0
                fuzzy_match = fuzz.ratio(str_input1, str_input2) / 100.0
                current_step_score += normalized_weights['action_input'] * (exact_match * 0.7 + fuzzy_match * 0.3)
                current_step_component_weight_sum += normalized_weights['action_input']

            # Compare 'observation'
            obs1 = step1.get('observation', '')
            obs2 = step2.get('observation', '')
            if normalized_weights.get('observation', 0) > 0:
                exact_match = 1.0 if str(obs1) == str(obs2) else 0.0
                fuzzy_match = fuzz.ratio(str(obs1), str(obs2)) / 100.0
                current_step_score += normalized_weights['observation'] * (exact_match * 0.7 + fuzzy_match * 0.3)
                current_step_component_weight_sum += normalized_weights['observation']
            
            if current_step_component_weight_sum > 0:
                step_scores.append(current_step_score / current_step_component_weight_sum)
            else:
                step_scores.append(0.0) # No weighted components in this step

        # Calculate average step score
        avg_step_score = np.mean(step_scores) if step_scores else 0.0

        # Compare 'final_answer'
        final_answer1 = pred_trajectory.get('final_answer', '')
        final_answer2 = gold_trajectory.get('final_answer', '')
        final_answer_score = 0.0
        if normalized_weights.get('final_answer', 0) > 0:
            exact_match = 1.0 if str(final_answer1) == str(final_answer2) else 0.0
            fuzzy_match = fuzz.ratio(str(final_answer1), str(final_answer2)) / 100.0
            final_answer_score = (exact_match * 0.7 + fuzzy_match * 0.3)

        # Combine scores based on overall weights
        total_weighted_score = 0.0
        if normalized_weights.get('action', 0) > 0 or normalized_weights.get('action_input', 0) > 0 or normalized_weights.get('observation', 0) > 0:
            total_weighted_score += avg_step_score * (normalized_weights.get('action', 0) + normalized_weights.get('action_input', 0) + normalized_weights.get('observation', 0))
        
        if normalized_weights.get('final_answer', 0) > 0:
            total_weighted_score += final_answer_score * normalized_weights['final_answer']
        
        return total_weighted_score

    def evaluate(self, pred_trajectory: Dict[str, Any], gold_trajectory: Dict[str, Any], weights: Optional[Dict[str, float]] = None):
        """
        Evaluates a predicted trajectory against a gold standard trajectory using a weighted overlap score.
        """
        # Pass weights to multi_step_overlap
        overall_overlap_score = self.multi_step_overlap(pred_trajectory, gold_trajectory, weights)
        
        # For backward compatibility or if other metrics are needed, they can be calculated here.
        # For this task, the user requested a single weighted score from multi_step_overlap.
        
        return {
            "OverallWeightedOverlapScore": overall_overlap_score
        }
