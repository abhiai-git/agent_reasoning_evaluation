import json
from typing import List, Dict, Any, Optional
from .providers import LLMProvider

class EvidenceBank:
    """
    A dynamically constructed knowledge base that stores factual information
    gathered by the agent throughout its trajectory.
    """
    def __init__(self):
        self.evidence: List[Dict[str, Any]] = []

    def add(self, step: Dict[str, Any]):
        """
        Adds a new piece of evidence (action, input, observation) to the bank.
        """
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
        For now, it performs a simple substring check against all collected evidence
        up to the given step_index.

        Args:
            claim: The thought/claim to verify.
            step_index: The 0-indexed step number for which the thought is being evaluated.
                        Evidence up to this index (exclusive) is considered.

        Returns:
            True if the claim is found in the evidence, False otherwise.
        """
        # For the first step (step_index 0), evidence_so_far will be empty.
        evidence_so_far = self.evidence[:step_index]
        return any(claim.lower() in json.dumps(e).lower() for e in evidence_so_far)


class TraceEvaluator:
    """
    Implements the TRACE framework for multi-dimensional evaluation of an agent's
    reasoning trajectory.
    """
    def __init__(self, provider: Optional[LLMProvider] = None):
        """
        Initializes the TraceEvaluator.

        Args:
            provider: An optional LLMProvider instance for interacting with an LLM
                      to perform evaluation tasks (e.g., grounding checks).
                      If not provided, metrics will use simplified heuristics.
        """
        self.provider = provider
        if not self.provider:
            print("Warning: LLMProvider not provided. TRACE metrics will use simplified heuristics.")

    def score_efficiency(self, evidence_bank: EvidenceBank) -> float:
        """
        Calculates efficiency: |E_min| / |E_n|.
        In a full TRACE implementation, an LLM would identify E_min (minimal necessary evidence).
        This is a placeholder using a fixed heuristic for demonstration.

        Args:
            evidence_bank: The EvidenceBank instance containing all collected evidence.

        Returns:
            A float representing the efficiency score (0.0 to 1.0).
        """
        total_evidence_count = len(evidence_bank.evidence)
        if total_evidence_count == 0:
            return 1.0 # Perfectly efficient if no steps were taken

        # Heuristic: Assume 60% of steps are necessary for demonstration purposes.
        # In a real scenario, this would be determined by an LLM.
        e_min = max(1, int(0.6 * total_evidence_count))
        return e_min / total_evidence_count

    def score_hallucination(self, trajectory: List[Dict[str, Any]], evidence_bank: EvidenceBank) -> float:
        """
        Calculates hallucination rate: sum(H(s_t)) / n.
        H(s_t) = 1 if thought is ungrounded (a hallucination).

        Args:
            trajectory: A list of dictionaries, where each dictionary represents a step
                        in the agent's reasoning process.
            evidence_bank: The EvidenceBank instance containing all collected evidence.

        Returns:
            A float representing the hallucination rate (0.0 to 1.0).
        """
        if not trajectory:
            return 0.0 # No hallucinations if no steps were taken

        ungrounded_count = 0
        for i, step in enumerate(trajectory):
            thought = step.get("thought", "")
            # Evaluate the thought against evidence collected *before* this step (E_{t-1} in paper)
            if not evidence_bank.verify_fact(thought, i): # Pass current step index for evidence context
                ungrounded_count += 1
        
        return ungrounded_count / len(trajectory)

    def score_adaptivity(self, trajectory: List[Dict[str, Any]]) -> float:
        """
        Calculates adaptivity score: sum(Adp(s_t+1)) / |F|.
        Adp(s_t+1) = 1 if agent adapts after a tool failure.

        Args:
            trajectory: A list of dictionaries representing the agent's trajectory.

        Returns:
            A float representing the adaptivity score (0.0 to 1.0).
        """
        if not trajectory:
            return 0.0

        failure_events = 0
        adaptive_responses = 0

        for t in range(len(trajectory) - 1):
            current_step = trajectory[t]
            next_step = trajectory[t+1]
            
            observation = current_step.get("observation", "").lower()
            
            # Check for tool execution failure (simplified heuristic)
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
                # Tries a different action OR
                # If the next step is a final answer after an error, it might also be adaptive
                if (("alternative" in subsequent_thought or "try another" in subsequent_thought) and \
                   (subsequent_action != failed_action)) or \
                   (next_step.get("final_answer") is not None and \
                    ("error" in subsequent_thought or "not available" in subsequent_thought or "failed" in subsequent_thought)):
                    adaptive_responses += 1

        if failure_events == 0:
            return 1.0 # Perfectly adaptive if no failures occurred
        
        return adaptive_responses / failure_events

    def score_instruction_error(self, trajectory: List[Dict[str, Any]], valid_tools: List[str]) -> float:
        """
        Calculates instruction error rate: count(invalid actions or input formats) / n.

        Args:
            trajectory: A list of dictionaries representing the agent's trajectory.
            valid_tools: A list of valid tool names.

        Returns:
            A float representing the instruction error rate (0.0 to 1.0).
        """
        if not trajectory:
            return 0.0

        invalid_count = 0
        for step in trajectory:
            action = step.get("action")
            # action_input = step.get("action_input") # Not currently used for heuristic check

            # Check for non-existent tool selection
            if action not in valid_tools:
                invalid_count += 1
            # Placeholder for checking incorrect argument format
            # In a full TRACE implementation, an LLM or schema validator would check action_input.
            # For now, this heuristic only checks for invalid tool names.
            # elif not self._is_valid_argument_format(action, action_input):
            #     invalid_count += 1
        
        return invalid_count / len(trajectory)

    def evaluate_trace(self, trajectory: List[Dict[str, Any]], valid_tools: List[str]) -> Dict[str, float]:
        """
        Evaluates a given trajectory using the TRACE metrics.

        Args:
            trajectory: A list of dictionaries representing the agent's trajectory.
            valid_tools: A list of valid tool names for instruction error checking.

        Returns:
            A dictionary containing the calculated TRACE metrics.
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
        
        # Efficiency is typically measured only for successful trajectories.
        # This implementation assumes the trajectory is "successful" for scoring purposes.
        efficiency_score = self.score_efficiency(eb)

        return {
            "Efficiency": efficiency_score,
            "Hallucination": self.score_hallucination(trajectory, eb),
            "Adaptivity": self.score_adaptivity(trajectory),
            "InstructionError": self.score_instruction_error(trajectory, valid_tools)
        }
