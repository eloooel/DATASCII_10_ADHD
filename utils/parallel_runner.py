from concurrent.futures import ProcessPoolExecutor, as_completed
import concurrent.futures
from tqdm import tqdm
import multiprocessing
from typing import List, Callable, Any, Dict
import os


def _safe_worker_wrapper(worker_fn: Callable[[Any], Any], task: Any) -> Dict[str, Any]:
    """
    Wrapper that safely imports necessary dependencies inside each worker process.
    """
    # Add these imports at the start of the function
    import torch
    import nibabel as nib
    import numpy as np
    
    # Initialize CUDA device for each worker
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(task, dict):
        task['device'] = device  # Pass device to the task
    
    try:
        result = worker_fn(task)
        return result
    except Exception as e:
        return {"status": "failed", "error": str(e), "task": str(task)}


def run_parallel(func, items, max_workers=None, desc="Processing"):
    """Run function in parallel with proper progress tracking"""
    if max_workers is None:
        max_workers = min(2, os.cpu_count() or 1)
    
    # Use ProcessPoolExecutor with as_completed for real-time progress
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_item = {executor.submit(func, item): item for item in items}
        
        # Track completed tasks
        results = []
        
        # Only show progress bar if desc is provided
        if desc:
            with tqdm(total=len(items), desc=desc) as pbar:
                for future in concurrent.futures.as_completed(future_to_item):
                    try:
                        result = future.result()
                        results.append(result)
                        pbar.update(1)
                    except Exception as e:
                        item = future_to_item[future]
                        print(f"Error processing {item}: {e}")
                        results.append({"status": "error", "error": str(e)})
                        pbar.update(1)
        else:
            # No progress bar - just collect results silently
            for future in concurrent.futures.as_completed(future_to_item):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    item = future_to_item[future]
                    print(f"Error processing {item}: {e}")
                    results.append({"status": "error", "error": str(e)})
    
    return results
