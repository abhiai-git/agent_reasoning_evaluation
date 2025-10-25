import os
import json
from agent_trajectory_evaluation import LLMConfig
from agent_trajectory_evaluation.unified import UnifiedEvaluator
from typing import List, Dict, Any

# --- LangChain Integration Example ---
# This example demonstrates how to capture a LangChain agent's trace
# and convert it into the format compatible with this evaluation package.

try:
    from langchain_openai import ChatOpenAI
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.tools import tool
    from langchain_core.messages import HumanMessage
    from langchain_core.tracers.run_collector import RunCollectorCallbackHandler
    from langchain_core.runnables import RunnableConfig
    from langchain_core.outputs import Run
except ImportError:
    print("LangChain dependencies not installed. Skipping LangChain example.")
    print("Please install with: pip install langchain langchain-openai langchain-community")
    ChatOpenAI = None # Set to None to skip the example

if ChatOpenAI:
    # 1. Define some simple tools for the LangChain agent
    @tool
    def multiply(a: int, b: int) -> int:
        """Multiplies two numbers."""
        return a * b

    @tool
    def add(a: int, b: int) -> int:
        """Adds two numbers."""
        return a + b

    tools = [multiply, add]

    # 2. Create a simple LangChain agent
    # The LLM is configured with an API key from environment variables.
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
    agent = create_tool_calling_agent(llm, tools)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    # 3. Function to parse LangChain trace into evaluation package format
    def parse_langchain_trace_to_steps(run: Run) -> List[Dict[str, Any]]:
        """
        Parses a LangChain Run object (trace) into a list of steps
        compatible with the evaluation package.

        Args:
            run: The LangChain Run object representing the agent's execution trace.

        Returns:
            A list of dictionaries, each representing a step in the evaluation package format.
        """
        parsed_steps = []
        final_answer = None

        # Collect all tool calls and their outputs from the LangChain trace
        tool_interactions = []
        for child_run in run.child_runs:
            if child_run.run_type == "tool":
                tool_name = child_run.name
                tool_input = child_run.inputs
                tool_output = child_run.outputs.get("output") if child_run.outputs else None
                tool_interactions.append({
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                    "tool_output": tool_output,
                    "start_time": child_run.start_time,
                    "end_time": child_run.end_time
                })
            elif child_run.run_type == "llm":
                # Look for agent thoughts and final answers from LLM calls
                # This parsing is a heuristic and might need refinement based on the specific
                # LangChain agent type and its output format.
                if child_run.outputs and child_run.outputs.get("generations"):
                    for generation in child_run.outputs["generations"]:
                        text = generation[0].get("text", "")
                        if "Final Answer:" in text:
                            final_answer = text.split("Final Answer:", 1)[1].strip()

        # Sort tool interactions by start time to reconstruct the chronological sequence
        tool_interactions.sort(key=lambda x: x["start_time"])

        # Reconstruct steps in the evaluation package format
        for i, interaction in enumerate(tool_interactions):
            # Simplified thought: assume the thought is to use the tool.
            # In a real scenario, you'd parse the LLM's actual thought from its output
            # if available and distinct from the action.
            thought = f"I need to use the {interaction['tool_name']} tool."
            parsed_steps.append({
                "thought": thought,
                "action": interaction["tool_name"],
                "action_input": interaction["tool_input"],
                "observation": interaction["tool_output"]
            })

        # Add the final answer as a separate step if found
        if final_answer:
            parsed_steps.append({"final_answer": final_answer})

        return parsed_steps

    # 4. Run the LangChain agent and capture its trace
    print("\n--- LangChain Agent Trace Example ---")
    cb = RunCollectorCallbackHandler() # Callback to collect the run trace
    cfg = RunnableConfig(callbacks=[cb]) # Configure the agent to use the callback

    # Ensure OPENAI_API_KEY environment variable is set
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY environment variable not set. Skipping LangChain example.")
    else:
        try:
            question = "What is 12 times 3 plus 5?"
            print(f"Running LangChain agent for: '{question}'")
            agent_executor.invoke({"input": question}, config=cfg) # Invoke the agent
            langchain_run = cb.traced_runs[0] # Get the captured run trace

            # 5. Parse the LangChain trace into the evaluation package format
            parsed_trajectory = parse_langchain_trace_to_steps(langchain_run)
            print("\nParsed Trajectory (from LangChain trace):")
            print(json.dumps(parsed_trajectory, indent=2))

            # 6. Evaluate the parsed trajectory using this evaluation package
            evaluator_config = LLMConfig(provider="openai", model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
            evaluator = UnifiedEvaluator(evaluator_config)
            
            # Define valid tools for InstructionError evaluation
            valid_tools_for_langchain = ["multiply", "add"]

            evaluation_results = evaluator.evaluate(parsed_trajectory, valid_tools=valid_tools_for_langchain)
            print("\nEvaluation Results (using this package):")
            for k, v in evaluation_results.items():
                print(f"  {k}: {v:.3f}")

        except Exception as e:
            print(f"An error occurred during LangChain example: {e}")
            print("Please ensure your OPENAI_API_KEY is set and LangChain dependencies are correctly installed.")
