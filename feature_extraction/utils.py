"""
Utility functions for parallel processing
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, List, Any

def run_parallel(func: Callable, items: List[Any], max_workers: int = None, desc: str = "Processing") -> List[Any]:
    """
    Run a function in parallel across multiple items
    
    Args:
        func: Function to execute
        items: List of items to process
        max_workers: Maximum number of parallel workers
        desc: Description for progress tracking
        
    Returns:
        List of results
    """
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(func, item): item for item in items}
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                item = futures[future]
                results.append({
                    "status": "failed",
                    "item": str(item),
                    "error": str(e)
                })
    
    return results
