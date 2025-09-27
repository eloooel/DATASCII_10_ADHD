from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing
from typing import List, Callable, Any
import sys

def run_parallel(
    tasks: List[Any],
    worker_fn: Callable,
    max_workers: int = None,
    desc: str = "Processing"
) -> List[Any]:
    """
    Run tasks in parallel with progress bar
    
    Args:
        tasks: List of items to process
        worker_fn: Function to process each task
        max_workers: Number of parallel workers (defaults to CPU count)
        desc: Description for progress bar
    """
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    results = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker_fn, task): i for i, task in enumerate(tasks)}

        for i, future in enumerate(tqdm(futures, total=len(futures), desc=desc)):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = {"status": "failed", "error": str(e)}
    return results