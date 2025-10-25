import json
from typing import List, Dict, Any, Optional
from .providers import LLMProvider

class EvidenceBank:
    def __init__(self):
        self.evidence: List[Dict[str, Any]] = []

    def add(self, step: Dict[str, Any]):
        # Store action, input, and observation as evidence
        self.evidence.append({
            "action": step.get("action"),
            "action_input": step.get("action_input"),
            "observation": step.get("observation")
        })

    def verify_fact(self, claim: str, step_index: int) -> bool:
        """
        Placeholder for LLM-based grounding check.
        In a full TRACE implementation, an LLM would determine if 'claim'
        is logically derivable from evidence up to 'step_index - 1'.
        For now, it performs a simple substring check against all collected evidence.
        """
        # For the first step, evidence_so_far is empty, so grounding check is different.
        # This simplified check just looks at all evidence.
        evidence_so_far = self.evidence[:step_index]
        return any(claim.lower() in json.dumps(e).lower() for e in evidence_so_far)


class TraceEvaluator:
    def __init__(self, provider: Optional[LLMProvider] = None):
        self.provider = provider
        if not self.provider:
            print("Warning: LLMProvider not provided. TRACE metrics will use simplified heuristics.")

    def score_efficiency(self, evidence_bank: EvidenceBank) -> float:
        """
        Calculates efficiency: |E_min| / |E_n|.
        In a full TRACE implementation, an LLM would identify E_min.
        This is a placeholder using a fixed heuristic.
        """
        total_evidence_count = len(evidence_bank.evidence)
        if total_evidence_count == 0:
            return 1.0
        # Heuristic: Assume 60% of steps are necessary for demonstration
        e_min = max(1, int(0.6 * total_evidence_count))
        return e_min / total_evidence_count

    def score_hallucination(self, trajectory: List[Dict[str, Any]], evidence_bank: EvidenceBank) -> float:
        """
        Calculates hallucination rate: sum(H(s_t)) / n.
        H(s_t) = 1 if thought is ungrounded.
        """
        if not trajectory:
            return 0.0 # No hallucinations if no steps

        ungrounded_count = 0
        for i, step in enumerate(trajectory):
            thought = step.get("thought", "")
            # For the first step, evidence_so_far is empty, so grounding check is different.
            # The verify_fact method needs to be aware of the step index.
            # Here, we pass i (0-indexed) which corresponds to evidence up to step i.
            # The paper uses E_{t-1} for th_t, so for step i, we check against evidence up to i.
            if not evidence_bank.verify_fact(thought, i): # Pass current step index for evidence context
                ungrounded_count += 1
        
        return ungrounded_count / len(trajectory)

    def score_adaptivity(self, trajectory: List[Dict[str, Any]]) -> float:
        """
        Calculates adaptivity score: sum(Adp(s_t+1)) / |F|.
        Adp(s_t+1) = 1 if agent adapts after tool failure.
        """
        if not trajectory:
            return 0.0

        failure_events = 0
        adaptive_responses = 0

        for t in range(len(trajectory) - 1):
            current_step = trajectory[t]
            next_step = trajectory[t+1]
            
            observation = current_step.get("observation", "").lower()
            
            # Check for tool execution failure
            if "error" in observation or "not available" in observation:
                failure_events += 1
                
                # Placeholder for LLM-based adaptivity check
                # In a full TRACE implementation, an LLM would assess if next_step's thought
                # acknowledges failure and action is a sensible alternative.
                subsequent_thought = next_step.get("thought", "").lower()
                subsequent_action = next_step.get("action", "")
                failed_action = current_step.get("action", "")

                # Simplified heuristic for adaptivity:
                # Acknowledges failure (simple keyword check) AND
                # Tries a different action
                if ("alternative" in subsequent_thought or "try another" in subsequent_thought) and \
                   (subsequent_action != failed_action):
                    adaptive_responses += 1
                # If the next step is a final answer after an error, it might also be adaptive
                elif next_step.get("final_answer") is not None and ("error" in subsequent_thought or "not available" in subsequent_thought):
                    adaptive_responses += 1

        if failure_events == 0:
            return 1.0 # Perfectly adaptive if no failures occurred
        
        return adaptive_responses / failure_events

    def score_instruction_error(self, trajectory: List[Dict[str, Any]], valid_tools: List[str]) -> float:
        """
        Calculates instruction error rate: count(invalid actions or input formats) / n.
        """
        if not trajectory:
            return 0.0

        invalid_count = 0
        for step in trajectory:
            action = step.get("action")
            action_input = step.get("action_input")

            # Check for non-existent tool selection
            if action not in valid_tools:
                invalid_count += 1
            # Placeholder for checking incorrect argument format
            # In a full TRACE implementation, an LLM or schema validator would check action_input.
            # For now, we assume valid if action is valid.
            # elif not self._is_valid_argument_format(action, action_input):
            #     invalid_count += 1
        
        return invalid_count / len(trajectory)

    def evaluate_trace(self, trajectory: List[Dict[str, Any]], valid_tools: List[str]) -> Dict[str, float]:
        """
        Evaluates a given trajectory using the TRACE metrics.
        """
        if not trajectory:
            return {
                "Efficiency": 0.0,
                "Hallucination": 0.0,
                "Adaptivity": 0.0,
                "InstructionError": 0.0
            }

        eb = EvidenceBank()
        for step in trajectory:
            # Only add steps that have an action and observation to the evidence bank
            # as per the paper's definition of evidence e_t = (a_t, i_t, o_t)
            if step.get("action") and step.get("observation") is not None:
                eb.add(step)
        
        # Efficiency is only measured for successful trajectories.
        # This implementation assumes the trajectory is "successful" for scoring purposes.
        efficiency_score = self.score_efficiency(eb)

        return {
            "Efficiency": efficiency_score,
            "Hallucination": self.score_hallucination(trajectory, eb),
            "Adaptivity": self.score_adaptivity(trajectory),
            "InstructionError": self.score_instruction_error(trajectory, valid_tools)
        }
