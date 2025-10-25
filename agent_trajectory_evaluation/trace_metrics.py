import json
from typing import List, Dict
from .providers import LLMProvider

class EvidenceBank:
    def __init__(self):
        self.evidence = []

    def add(self, step):
        self.evidence.append({
            "action": step.get("action"),
            "input": step.get("action_input"),
            "observation": step.get("observation")
        })

    def verify_fact(self, claim: str) -> bool:
        return any(claim.lower() in json.dumps(e).lower() for e in self.evidence)


class TraceEvaluator:
    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def score_efficiency(self, evidence_bank: EvidenceBank):
        e_min = max(1, int(0.6 * len(evidence_bank.evidence)))
        return e_min / len(evidence_bank.evidence)

    def score_hallucination(self, steps: List[Dict], evidence_bank: EvidenceBank):
        ungrounded = 0
        for s in steps:
            if not evidence_bank.verify_fact(s["thought"]):
                ungrounded += 1
        return 1 - (ungrounded / len(steps))

    def score_adaptivity(self, steps: List[Dict]):
        adaptive = 0
        for t in range(len(steps) - 1):
            if "error" in steps[t]["observation"].lower():
                if steps[t + 1]["action"] != steps[t]["action"]:
                    adaptive += 1
        return adaptive / len(steps)

    def score_instruction_error(self, steps: List[Dict], valid_tools: List[str]):
        invalid = sum(1 for s in steps if s["action"] not in valid_tools)
        return 1 - (invalid / len(steps))

    def evaluate_trace(self, trajectory, valid_tools: List[str]):
        eb = EvidenceBank()
        for s in trajectory:
            eb.add(s)
        return {
            "Efficiency": self.score_efficiency(eb),
            "Hallucination": self.score_hallucination(trajectory, eb),
            "Adaptivity": self.score_adaptivity(trajectory),
            "InstructionError": self.score_instruction_error(trajectory, valid_tools)
        }