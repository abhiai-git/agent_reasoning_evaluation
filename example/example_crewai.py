import os
import json
from agent_trajectory_evaluation import LLMConfig
from agent_trajectory_evaluation.unified import UnifiedEvaluator
from typing import List, Dict, Any

# --- CrewAI Integration Example ---
# This example demonstrates how to simulate a CrewAI agent's trace
# and convert it into the format compatible with this evaluation package.

# For a real CrewAI integration, you would use the actual CrewAI framework
# to define your agents, tasks, and crew, then execute it to capture its trace.
# This example provides a simplified simulation of a CrewAI-like interaction
# and a parser for such a trace.

try:
    from crewai import Agent, Task, Crew, Process
    from crewai_tools import BaseTool
    from langchain_openai import ChatOpenAI # CrewAI often uses LangChain LLMs
except ImportError:
    print("CrewAI dependencies not installed. Skipping CrewAI example.")
    print("Please install with: pip install crewai crewai-tools langchain-openai")
    Agent = None # Set to None to skip the example

if Agent:
    # 1. Define some simple tools for the CrewAI agents
    class CalculatorTool(BaseTool):
        name: str = "Calculator"
        description: str = "Useful for performing arithmetic operations."

        def _run(self, operation: str) -> str:
            """Performs a simple arithmetic operation."""
            try:
                return str(eval(operation))
            except Exception as e:
                return f"Error: {e}"

    class SearchTool(BaseTool):
        name: str = "SearchTool"
        description: str = "Useful for searching information on the internet."

        def _run(self, query: str) -> str:
            """Simulates a web search."""
            if "current year" in query.lower():
                return "The current year is 2025."
            return f"Search results for '{query}': Found relevant information."

    calculator_tool = CalculatorTool()
    search_tool = SearchTool()
    crew_tools = [calculator_tool, search_tool]

    # 2. Simulate a CrewAI agent's interaction trace
    # In a real CrewAI run, you would execute a crew and capture its logs/intermediate steps.
    # This simulation mimics the structure of such an interaction.
    def simulate_crewai_trace(question: str) -> List[Dict[str, Any]]:
        """
        Simulates a simplified CrewAI agent's interaction trace for a given question.
        This trace will contain 'thought', 'action', 'action_input', 'observation',
        and potentially 'final_answer'.
        """
        trace = []
        final_answer = None

        if "what is the current year plus 10" in question.lower():
            # Agent 1: Search for current year
            thought1 = "I need to find the current year first using the SearchTool."
            action1 = "SearchTool"
            input1 = {"query": "current year"}
            observation1 = search_tool._run("current year")
            trace.append({
                "thought": thought1,
                "action": action1,
                "action_input": input1,
                "observation": observation1
            })

            # Agent 2: Calculate current year + 10
            thought2 = f"The current year is 2025. I need to add 10 to it using the Calculator."
            action2 = "Calculator"
            input2 = {"operation": "2025 + 10"}
            observation2 = calculator_tool._run("2025 + 10")
            trace.append({
                "thought": thought2,
                "action": action2,
                "action_input": input2,
                "observation": observation2
            })
            
            final_answer = str(observation2)
            trace.append({"final_answer": final_answer})

        return trace

    # 3. Function to parse CrewAI trace into evaluation package format
    # (In this simulated example, the trace is already in a compatible format,
    # but a real parser would extract from CrewAI-specific objects/logs)
    def parse_crewai_trace_to_steps(crewai_trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Parses a simulated CrewAI trace (which is already close to the target format)
        into the exact list of steps compatible with the evaluation package.

        Args:
            crewai_trace: A list of dictionaries representing the simulated CrewAI trace.

        Returns:
            A list of dictionaries, each representing a step in the evaluation package format.
        """
        # For this simulation, we assume the 'simulate_crewai_trace' function
        # already produces a list of dictionaries that are almost directly usable.
        # We just need to ensure the 'final_answer' is at the end as a separate step
        # and other steps have 'thought', 'action', 'action_input', 'observation'.
        
        parsed_steps = []
        final_answer_step = None

        for step in crewai_trace:
            if "final_answer" in step:
                final_answer_step = {"final_answer": step["final_answer"]}
            else:
                parsed_steps.append(step)
            
        if final_answer_step:
            parsed_steps.append(final_answer_step)

        return parsed_steps

    # 4. Run the simulated CrewAI agent and capture trace
    print("\n--- CrewAI Agent Trace Example ---")

    question = "What is the current year plus 10?"
    print(f"Simulating CrewAI agent for: '{question}'")
    simulated_crewai_trace = simulate_crewai_trace(question)

    # 5. Parse the simulated CrewAI trace
    parsed_trajectory = parse_crewai_trace_to_steps(simulated_crewai_trace)
    print("\nParsed Trajectory (from simulated CrewAI trace):")
    print(json.dumps(parsed_trajectory, indent=2))

    # 6. Evaluate the parsed trajectory using the evaluation package
    # Ensure OPENAI_API_KEY is set in environment for actual LLM evaluation
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY environment variable not set. Skipping CrewAI evaluation with LLM.")
    else:
        try:
            evaluator_config = LLMConfig(provider="openai", model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
            evaluator = UnifiedEvaluator(evaluator_config)
            
            # Define valid tools for InstructionError evaluation
            valid_tools_for_crewai = ["Calculator", "SearchTool"]

            evaluation_results = evaluator.evaluate(parsed_trajectory, valid_tools=valid_tools_for_crewai)
            print("\nEvaluation Results (using this package):")
            for k, v in evaluation_results.items():
                print(f"  {k}: {v:.3f}")

        except Exception as e:
            print(f"An error occurred during CrewAI example evaluation: {e}")
            print("Please ensure your OPENAI_API_KEY is set and CrewAI dependencies are correctly installed.")
