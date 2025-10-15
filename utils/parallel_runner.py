from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing
from typing import List, Callable, Any, Dict


def _safe_worker_wrapper(worker_fn: Callable[[Any], Any], task: Any) -> Dict[str, Any]:
    """
    Wrapper that safely imports necessary dependencies inside each worker process.
    Prevents NameError issues (e.g., 'np' undefined) when multiprocessing reloads the module.
    """
    # Local imports for isolation
    import numpy as np  # ensures np exists in all child processes
    import torch        # optional but useful if preprocessing uses torch tensors
    import nibabel as nib
    import os

    try:
        result = worker_fn(task)
        return result
    except Exception as e:
        return {"status": "failed", "error": str(e), "task": str(task)}


def run_parallel(tasks: List[Any], worker_fn: Callable[[Any], Any], max_workers: int = None) -> List[Any]:
    """
    Runs a list of tasks in parallel using ProcessPoolExecutor with a shared progress bar.
    Each worker runs in isolation with safe imports to prevent NameError issues.
    """
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    results = [None] * len(tasks)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_safe_worker_wrapper, worker_fn, task): i
            for i, task in enumerate(tasks)
        }

        with tqdm(total=len(futures), desc="Preprocessing subjects") as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = {"status": "failed", "error": str(e)}
                pbar.update(1)

    return results
