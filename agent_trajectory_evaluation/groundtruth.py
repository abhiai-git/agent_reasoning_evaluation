import numpy as np
import difflib
from typing import Dict, List, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer, util
from fuzzywuzzy import fuzz
import json

class GroundTruthEvaluator:
    """
    Evaluates a predicted agent trajectory against a ground truth trajectory
    using various matching strategies, including exact, fuzzy, and weighted overlap.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the GroundTruthEvaluator with a SentenceTransformer model for semantic matching.

        Args:
            model_name: The name of the SentenceTransformer model to use for semantic fuzzy matching.
        """
        self.embedder = SentenceTransformer(model_name)

    @staticmethod
    def exact_match(pred: str, gold: str) -> float:
        """
        Performs an exact, case-insensitive match between two strings.

        Args:
            pred: The predicted string.
            gold: The ground truth string.

        Returns:
            1.0 if strings match exactly (case-insensitive, stripped), 0.0 otherwise.
        """
        return 1.0 if pred.strip().lower() == gold.strip().lower() else 0.0

    def fuzzy_match(self, pred: str, gold: str, method: str = "semantic") -> float:
        """
        Performs a fuzzy match between two strings.

        Args:
            pred: The predicted string.
            gold: The ground truth string.
            method: The fuzzy matching method to use ('semantic' or 'token').

        Returns:
            A float score (0.0 to 1.0) indicating similarity.
        """
        if method == "semantic":
            # Encode strings into embeddings and calculate cosine similarity
            emb_pred = self.embedder.encode(pred, convert_to_tensor=True)
            emb_gold = self.embedder.encode(gold, convert_to_tensor=True)
            return float(util.cos_sim(emb_pred, emb_gold))
        elif method == "token":
            # Use fuzzywuzzy's token ratio for token-based fuzzy matching
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
        # Merge default weights with any provided custom weights
        component_weights = {**default_weights, **(weights if weights is not None else {})}
        
        # Normalize weights if they don't sum to 1, to ensure scores are comparable
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
            
            # Special handling for action_input which is a dictionary
            if component_name == 'action_input':
                str_val1 = json.dumps(val1, sort_keys=True) # Convert to sorted JSON string for consistent comparison
                str_val2 = json.dumps(val2, sort_keys=True)
            else:
                str_val1 = str(val1)
                str_val2 = str(val2)

            exact_match = 1.0 if str_val1 == str_val2 else 0.0
            fuzzy_match = fuzz.ratio(str_val1, str_val2) / 100.0
            # Combine exact and fuzzy match with a fixed ratio (e.g., 70% exact, 30% fuzzy)
            return (exact_match * 0.7 + fuzzy_match * 0.3)

        def calculate_step_overall_score(step1: Dict, step2: Dict) -> Tuple[float, float]:
            """Calculates the weighted score for a single step and its total component weight."""
            current_step_score = 0.0
            current_step_component_weight_sum = 0.0
            # Iterate over relevant components for a step
            for comp in ['action', 'action_input', 'observation']:
                if normalized_weights.get(comp, 0) > 0: # Only consider if component has a weight
                    current_step_score += normalized_weights[comp] * calculate_component_score(step1, step2, comp)
                    current_step_component_weight_sum += normalized_weights[comp]
            return current_step_score, current_step_component_weight_sum

        if trajectory_match_mode == "ordered":
            # In ordered mode, compare steps one-to-one by their index
            min_len = min(len(pred_steps), len(gold_steps))
            for i in range(min_len):
                score, weight_sum = calculate_step_overall_score(pred_steps[i], gold_steps[i])
                if weight_sum > 0:
                    step_scores.append(score / weight_sum) # Normalize individual step score
                else:
                    step_scores.append(0.0) # No weighted components in this step
        else: # "unordered" mode
            # In unordered mode, find the best matching gold step for each predicted step
            matched_gold_indices = set() # Keep track of gold steps already matched
            for pred_step in pred_steps:
                best_match_score = 0.0
                best_match_gold_idx = -1
                
                for gold_idx, gold_step in enumerate(gold_steps):
                    if gold_idx not in matched_gold_indices: # Only consider unmatched gold steps
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
                    step_scores.append(0.0) # No good match found for this predicted step

        # Calculate average score across all matched/compared steps
        avg_step_score = np.mean(step_scores) if step_scores else 0.0

        # Compare 'final_answer' separately
        final_answer1 = pred_trajectory.get('final_answer', '')
        final_answer2 = gold_trajectory.get('final_answer', '')
        final_answer_score_val = 0.0
        if normalized_weights.get('final_answer', 0) > 0:
            final_answer_score_val = calculate_component_score({"final_answer": final_answer1}, {"final_answer": final_answer2}, "final_answer")

        # Combine scores based on overall weights
        total_weighted_score = 0.0
        # Sum of weights for step-level components
        step_components_total_weight = (
            normalized_weights.get('action', 0) +
            normalized_weights.get('action_input', 0) +
            normalized_weights.get('observation', 0)
        )
        
        # Add weighted average step score if step components have weight
        if step_components_total_weight > 0:
            total_weighted_score += avg_step_score * step_components_total_weight
        
        # Add weighted final answer score if final_answer has weight
        if normalized_weights.get('final_answer', 0) > 0:
            total_weighted_score += final_answer_score_val * normalized_weights['final_answer']
        
        return total_weighted_score

    def evaluate(self, pred_trajectory: Dict[str, Any], gold_trajectory: Dict[str, Any], weights: Optional[Dict[str, float]] = None, trajectory_match_mode: str = "ordered"):
        """
        Evaluates a predicted trajectory against a gold standard trajectory using a weighted overlap score.

        Args:
            pred_trajectory: The predicted trajectory.
            gold_trajectory: The ground truth trajectory.
            weights: Optional custom weights for components.
            trajectory_match_mode: 'ordered' or 'unordered' matching for steps.

        Returns:
            A dictionary containing the 'OverallWeightedOverlapScore'.
        """
        # Pass weights and trajectory_match_mode to multi_step_overlap for the main calculation
        overall_overlap_score = self.multi_step_overlap(pred_trajectory, gold_trajectory, weights, trajectory_match_mode)
        
        return {
            "OverallWeightedOverlapScore": overall_overlap_score
        }
