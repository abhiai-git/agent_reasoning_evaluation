import asyncio
from tqdm.asyncio import tqdm_asyncio

async def async_eval_one(evaluator, trajectory, valid_tools):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, evaluator.evaluate_trace, trajectory, valid_tools)

async def batch_evaluate(evaluator, trajectories, valid_tools):
    tasks = [asyncio.create_task(async_eval_one(evaluator, traj, valid_tools)) for traj in trajectories]
    results = await tqdm_asyncio.gather(*tasks)
    return results
