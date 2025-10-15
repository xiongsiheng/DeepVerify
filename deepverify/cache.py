#!/usr/bin/env python
"""
    deepverify.cache
    
    Utilities for caching, async, etc
"""

import os
import inspect
import pickle
import hashlib
import asyncio
from functools import wraps
from rich import print as rprint
from deepverify import config

def disk_cache(cache_dir='./.cache/search', verbose=False, ignore_fields=None):
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
            if not config.CACHE_ENABLE:
                if verbose:
                    rprint(f"[yellow]disk_cache: Disabled, running function {func.__name__}")
                return await func(*args, **kwargs)
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
            if not config.CACHE_ENABLE:
                if verbose:
                    rprint(f"[yellow]disk_cache: Disabled, running function {func.__name__}")
                return func(*args, **kwargs)
                
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
                    out = pickle.load(open(cache_path, 'rb'))
                    if verbose:
                        rprint(f"[green]disk_cache: Loaded from cache[/green] {cache_path}")
                    return out
                except Exception as e:
                    rprint(f"[red]disk_cache: Error loading cache: {cache_dir} {cache_path} {e}[/red]")
            else:
                if verbose:
                    rprint(f"[yellow]disk_cache: No cache found[/yellow] {cache_dir} {cache_path} - Running")
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

def disk_cache_fn(fn, fn_name=None, *d_args, **d_kwargs):
    if asyncio.iscoroutinefunction(fn):
        @disk_cache(*d_args, **d_kwargs)
        async def _fn(*args, **kwargs):
            return await fn(*args, **kwargs)
    else:
        @disk_cache(*d_args, **d_kwargs)
        def _fn(*args, **kwargs):
            return fn(*args, **kwargs)
    
    if fn_name is not None:
        _fn.__name__ = fn_name
    
    return _fn