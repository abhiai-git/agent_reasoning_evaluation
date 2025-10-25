import numpy as np
import difflib
from typing import Dict, List, Any
from sentence_transformers import SentenceTransformer, util

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
            return difflib.SequenceMatcher(None, pred.lower(), gold.lower()).ratio()
        else:
            raise ValueError("method must be 'semantic' or 'token'")

    def multi_step_overlap(self, pred_steps: List[Dict], gold_steps: List[Dict]) -> Dict[str, float]:
        total = len(gold_steps)
        exact_hits, fuzzy_hits = 0, 0
        for ps, gs in zip(pred_steps, gold_steps):
            if (ps["action"] == gs["action"]) and (ps["action_input"] == gs["action_input"]):
                exact_hits += 1
            else:
                ps_text = f"{ps['action']} {ps['action_input']}"
                gs_text = f"{gs['action']} {gs['action_input']}"
                fuzzy_hits += self.fuzzy_match(ps_text, gs_text, method="token") > 0.8
        return {
            "ExactStepMatch": exact_hits / total if total else 0.0,
            "FuzzyStepOverlap": fuzzy_hits / total if total else 0.0,
            "Composite": np.mean([
                exact_hits / total if total else 0.0,
                fuzzy_hits / total if total else 0.0
            ])
        }

    def evaluate(self, pred_trajectory: Dict[str, Any], gold_trajectory: Dict[str, Any]):
        pred_steps = pred_trajectory.get("steps", [])
        gold_steps = gold_trajectory.get("steps", [])
        traj_metrics = self.multi_step_overlap(pred_steps, gold_steps)
        exact_final = self.exact_match(pred_trajectory.get("final_answer", ""), gold_trajectory.get("final_answer", ""))
        fuzzy_final = self.fuzzy_match(pred_trajectory.get("final_answer", ""), gold_trajectory.get("final_answer", ""))
        return {
            "ExactFinalMatch": exact_final,
            "FuzzyFinalMatch": fuzzy_final,
            "TrajectoryMetrics": traj_metrics,
            "OverallScore": np.mean([traj_metrics["Composite"], fuzzy_final])
        }