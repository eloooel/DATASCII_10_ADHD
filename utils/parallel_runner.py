from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing
from typing import List, Callable, Any
import sys

def run_parallel(tasks, worker_fn, max_workers=None):
    if max_workers is None:
        import multiprocessing
        max_workers = multiprocessing.cpu_count()

    results = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker_fn, task): i for i, task in enumerate(tasks)}

        # Single progress bar in main process
        with tqdm(total=len(futures), desc="Preprocessing subjects") as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = {"status": "failed", "error": str(e)}
                pbar.update(1)

    return results