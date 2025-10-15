from functools import wraps
import hashlib
import os
import pickle
import inspect
import asyncio
from rich import print as rprint

def disk_cache(cache_dir='./.cache/search', verbose=True, ignore_fields=None):
    """
    Decorator that caches function results to disk.
    Works with both synchronous and asynchronous functions.
    
    Args:
        cache_dir: Directory to store cache files
        verbose: Whether to print cache status messages
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get cache key and path
            cache_str, cache_path = _get_cache_info(func, args, kwargs)
            
            # Return cached result if it exists
            cached_result = _try_get_cached_result(cache_path, cache_str, verbose)
            if cached_result is not None:
                return cached_result
                
            # Calculate result and cache it
            result = await func(*args, **kwargs)
            _save_to_cache(result, cache_path, cache_str, verbose)
            return result
            
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Get cache key and path
            cache_str, cache_path = _get_cache_info(func, args, kwargs)
            
            # Return cached result if it exists
            cached_result = _try_get_cached_result(cache_path, cache_str, verbose)
            if cached_result is not None:
                return cached_result
                
            # Calculate result and cache it
            result = func(*args, **kwargs)
            _save_to_cache(result, cache_path, cache_str, verbose)
            return result
        
        def _get_cache_info(func, args, kwargs):
            # Get the function signature
            sig = inspect.signature(func)
            
            # Create a dictionary of all parameters with their values
            params = {}
            
            # First, fill in with default values
            for param_name, param in sig.parameters.items():
                if param.default is not param.empty:
                    params[param_name] = param.default
            
            # Then update with positional arguments
            positional_params = list(sig.parameters.keys())
            for i, arg in enumerate(args):
                if i < len(positional_params):
                    params[positional_params[i]] = arg
            
            # Finally update with keyword arguments
            params.update(kwargs)
            
            if ignore_fields:
                for field in ignore_fields:
                    if field in params:
                        del params[field]
            
            cache_str = '-> '.join([func.__name__, str(sorted(params.items()))])
            cache_key = hashlib.md5(''.join(cache_str).encode()).hexdigest()
            cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")
            
            return cache_str, cache_path
        
        def _try_get_cached_result(cache_path, cache_str, verbose):
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'rb') as f:
                        out = pickle.load(f)
                        if verbose:
                            rprint(f"[green]disk_cache: Loaded from cache[/green] {cache_str}")
                        return out
                except Exception as e:
                    rprint(f"[red]disk_cache: Error loading cache: {cache_str} {e}[/red]")
            else:
                if verbose:
                    rprint(f"[yellow]disk_cache: No cache found[/yellow] {cache_str} - Running")
            return None
        
        def _save_to_cache(result, cache_path, cache_str, verbose):
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(result, f)
            except Exception as e:
                rprint(f"[red]disk_cache: Error saving to cache: {cache_str} {e}[/red]")
        
        # Return appropriate wrapper based on whether the function is async or not
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator
