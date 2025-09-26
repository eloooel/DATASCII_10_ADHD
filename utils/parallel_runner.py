from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Callable, Any
from tqdm import tqdm

def run_parallel(
    tasks: List[Any],
    worker_fn: Callable,
    max_workers: int = None,
    desc: str = "Running tasks"
) -> List[Any]:
    """
    Run tasks in parallel with a progress bar.

    Args:
        tasks (List[Any]): List of items to process (e.g., subjects).
        worker_fn (Callable): Function to apply to each item.
        max_workers (int, optional): Number of workers (defaults to os.cpu_count()).
        desc (str): Description for tqdm progress bar.

    Returns:
        List[Any]: List of results from worker_fn.
    """
    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker_fn, task): task for task in tasks}

        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            task = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append({"status": "failed", "error": str(e), "task": task})

    return results
