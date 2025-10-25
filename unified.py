from .config import LLMConfig
from .providers import LLMProvider
from .trace_metrics import TraceEvaluator
from .groundtruth import GroundTruthEvaluator

class UnifiedEvaluator:
    def __init__(self, config: LLMConfig, fuzzy_model: str = "all-MiniLM-L6-v2"):
        self.provider = LLMProvider(
            provider=config.provider,
            model=config.model,
            api_key=config.api_key,
            aws_profile=config.aws_profile
        )
        self.trace = TraceEvaluator(self.provider)
        self.groundtruth = GroundTruthEvaluator(model_name=fuzzy_model)

    def evaluate(self, trajectory, valid_tools, reference=None):
        trace_scores = self.trace.evaluate_trace(trajectory, valid_tools)
        if reference:
            gt_scores = self.groundtruth.evaluate(
                {"steps": trajectory, "final_answer": trajectory[-1].get("observation", "")},
                reference
            )
            return {**trace_scores, **gt_scores}
        return trace_scores
