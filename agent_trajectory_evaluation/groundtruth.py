import numpy as np
import difflib
from typing import Dict, List, Any, Optional, Tuple
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
        weights: Optional[Dict[str, float]] = None,
        trajectory_match_mode: str = "ordered"
    ) -> float:
        """
        Calculates a weighted overlap score between two trajectories, considering
        exact and fuzzy matches for action, action_input, observation, and final_answer.
        Supports 'ordered' and 'unordered' matching modes for trajectory steps.

        Args:
            pred_trajectory: A dictionary representing the predicted trajectory,
                             including 'steps' (list of dicts) and 'final_answer'.
            gold_trajectory: A dictionary representing the ground truth trajectory.
            weights: A dictionary of weights for different components.
                     Keys: 'action', 'action_input', 'observation', 'final_answer'.
                     Values: float (sum should ideally be 1.0 for normalized scores).
                     Default weights are used if not provided.
            trajectory_match_mode: 'ordered' to match steps by index, 'unordered' to find
                                   the best match for each predicted step in the gold trajectory.

        Returns:
            A float representing the weighted overlap score (0.0 to 1.0).
        """
        if trajectory_match_mode not in ["ordered", "unordered"]:
            raise ValueError("trajectory_match_mode must be 'ordered' or 'unordered'")

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

        step_scores = []
        
        def calculate_component_score(step1: Dict, step2: Dict, component_name: str) -> float:
            """Calculates the weighted exact/fuzzy score for a single component."""
            val1 = step1.get(component_name, '')
            val2 = step2.get(component_name, '')
            
            if component_name == 'action_input':
                str_val1 = json.dumps(val1, sort_keys=True)
                str_val2 = json.dumps(val2, sort_keys=True)
            else:
                str_val1 = str(val1)
                str_val2 = str(val2)

            exact_match = 1.0 if str_val1 == str_val2 else 0.0
            fuzzy_match = fuzz.ratio(str_val1, str_val2) / 100.0
            return (exact_match * 0.7 + fuzzy_match * 0.3)

        def calculate_step_overall_score(step1: Dict, step2: Dict) -> Tuple[float, float]:
            """Calculates the weighted score for a single step and its total component weight."""
            current_step_score = 0.0
            current_step_component_weight_sum = 0.0
            for comp in ['action', 'action_input', 'observation']:
                if normalized_weights.get(comp, 0) > 0:
                    current_step_score += normalized_weights[comp] * calculate_component_score(step1, step2, comp)
                    current_step_component_weight_sum += normalized_weights[comp]
            return current_step_score, current_step_component_weight_sum

        if trajectory_match_mode == "ordered":
            min_len = min(len(pred_steps), len(gold_steps))
            for i in range(min_len):
                score, weight_sum = calculate_step_overall_score(pred_steps[i], gold_steps[i])
                if weight_sum > 0:
                    step_scores.append(score / weight_sum)
                else:
                    step_scores.append(0.0)
        else: # "unordered" mode
            matched_gold_indices = set()
            for pred_step in pred_steps:
                best_match_score = 0.0
                best_match_gold_idx = -1
                
                for gold_idx, gold_step in enumerate(gold_steps):
                    if gold_idx not in matched_gold_indices:
                        score, weight_sum = calculate_step_overall_score(pred_step, gold_step)
                        if weight_sum > 0:
                            normalized_score = score / weight_sum
                            if normalized_score > best_match_score:
                                best_match_score = normalized_score
                                best_match_gold_idx = gold_idx
                
                if best_match_gold_idx != -1:
                    step_scores.append(best_match_score)
                    matched_gold_indices.add(best_match_gold_idx)
                else:
                    step_scores.append(0.0) # No good match found for this pred_step

        # Calculate average step score
        avg_step_score = np.mean(step_scores) if step_scores else 0.0

        # Compare 'final_answer'
        final_answer1 = pred_trajectory.get('final_answer', '')
        final_answer2 = gold_trajectory.get('final_answer', '')
        final_answer_score_val = 0.0
        if normalized_weights.get('final_answer', 0) > 0:
            final_answer_score_val = calculate_component_score({"final_answer": final_answer1}, {"final_answer": final_answer2}, "final_answer")

        # Combine scores based on overall weights
        total_weighted_score = 0.0
        step_components_total_weight = (
            normalized_weights.get('action', 0) +
            normalized_weights.get('action_input', 0) +
            normalized_weights.get('observation', 0)
        )
        
        if step_components_total_weight > 0:
            total_weighted_score += avg_step_score * step_components_total_weight
        
        if normalized_weights.get('final_answer', 0) > 0:
            total_weighted_score += final_answer_score_val * normalized_weights['final_answer']
        
        return total_weighted_score

    def evaluate(self, pred_trajectory: Dict[str, Any], gold_trajectory: Dict[str, Any], weights: Optional[Dict[str, float]] = None, trajectory_match_mode: str = "ordered"):
        """
        Evaluates a predicted trajectory against a gold standard trajectory using a weighted overlap score.
        """
        # Pass weights and trajectory_match_mode to multi_step_overlap
        overall_overlap_score = self.multi_step_overlap(pred_trajectory, gold_trajectory, weights, trajectory_match_mode)
        
        return {
            "OverallWeightedOverlapScore": overall_overlap_score
        }
