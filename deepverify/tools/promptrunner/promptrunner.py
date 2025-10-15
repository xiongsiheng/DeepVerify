#!/usr/bin/env python
"""
    promptrunner.py
"""

import litellm
litellm.suppress_debug_info = True

import sys
import json
from hashlib import md5
from pathlib import Path
from pydantic import BaseModel
from typing import Optional, Callable
from litellm import completion, acompletion
from rich import print as rprint
from tenacity import retry, stop_any, stop_after_attempt, stop_after_delay, wait_exponential, retry_if_exception_type

from . import utils

def _cache_key(*args):
    h = ''
    for x in args:
        if isinstance(x, dict):
            h += md5(json.dumps(x, sort_keys=True).encode()).hexdigest()
        else:
            h += md5(str(x).encode()).hexdigest()
    
    return md5(h.encode()).hexdigest()

# --
# Futures

def aretry_wrapper(fn, n_retries, verbose):
    @retry(
        stop         = stop_any(stop_after_attempt(n_retries + 1), stop_after_delay(60)),
        wait         = wait_exponential(multiplier=2, min=1, max=16),
        retry        = retry_if_exception_type(Exception),
        reraise      = True,
        before_sleep = lambda retry_state: rprint(f"[yellow]aretry_wrapper: Retry {retry_state.attempt_number}/{n_retries}[/yellow]", file=sys.stderr) if verbose else None
    )
    async def __retry_fn(*args, **kwargs):
        return await fn(*args, **kwargs)
    return __retry_fn


class Future:
    def __init__(self, fn=None, value=None, n_retries=5, verbose=False):
        self.fn        = fn
        self.value     = value
        self.is_done   = value is not None
        self.n_retries = n_retries
        self.verbose   = verbose
    
    async def __call__(self):
        if self.is_done:
            return self.value
        
        if self.fn is None:
            raise Exception("Future has no function to call")
        
        self.value   = await aretry_wrapper(self.fn, n_retries=self.n_retries, verbose=self.verbose)()
        self.is_done = True
        return self.value

# --
# Main

class PromptRunner:
    LOG_DIR         = None
    DEFAULT_SYSTEM = "You are a helpful assistant"

    @classmethod
    def set_default_logdir(cls, path: str):
        import os
        cls.LOG_DIR = path
        os.makedirs(path, exist_ok=True)
    
    def __init__(
            self, 
            *, 
            name            : str,
            system          : Optional[str]         = None, 
            template        : str, 
            before          : Optional[Callable]    = None, 
            after           : Optional[Callable]    = None, 
            validate        : Optional[Callable]    = None,
            response_format : Optional[BaseModel]   = None,
            llm_kwargs      : dict,
            cache_dir       : Optional[str]         = None,
            no_log          : bool                  = False,
            no_console      : bool                  = False,
            cache_debug     : bool                  = False
        ):
        """
            name            : name of prompt
            system          : system prompt
            template        : template for user prompt
            before          : data preprocessing - gets run before prompt
            after           : post-processing - gets run after prompt
            validate        : validation - gets run after post-processing - should raise error if output is invalid
            response_format : response format - only supported by some models
            llm_kwargs      : kwargs for llm
            cache_dir       : cache directory
            no_log          : disable logging?
            no_console      : disable console output?
            cache_debug     : enable cache debugging?
        """
        self.name            = name
        self.counter         = 0
        self.system          = system if system else self.DEFAULT_SYSTEM
        self.template        = template
        self.response_format = response_format
        self.before          = before
        self.after           = after
        self.validate        = validate
        self.llm_kwargs      = llm_kwargs
        self.do_log          = not no_log
        self.do_console      = not no_console
        self.cache_debug     = cache_debug
        
        self.cache_dir = None
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir) / self.name
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def prompt(self, **inputs):
        if self.before is not None:
            prompt_kwargs = self.before(**inputs)
            extra_kwargs  = {k:v for k,v in prompt_kwargs.items() if k.startswith('__')}    # `before` inputs w/ __ are not passed to prompt & get forwarded to `after`
            prompt_kwargs = {k:v for k,v in prompt_kwargs.items() if k not in extra_kwargs}
        else:
            extra_kwargs  = {}
            prompt_kwargs = inputs
        
        prompt = self.template.format(**prompt_kwargs)
        return prompt, extra_kwargs
    
    def try_cache(self, _cache_idx=None, _cache_only=False, **inputs):
        # Format prompt
        prompt, _ = self.prompt(**inputs)
        
        # Read cache (?)
        cache_key = None
        if self.cache_dir is not None:
            if _cache_idx is None:
                cache_key = _cache_key(self.llm_kwargs, self.system, prompt)
            else:
                cache_key = _cache_key(self.llm_kwargs, self.system, prompt, _cache_idx)
            
            cache_path = Path(self.cache_dir) / f"{cache_key}.json"
            if cache_path.exists():
                if self.cache_debug:
                    rprint(f"[green]promptrunner {self.name}: Loaded from cache[/green] {cache_key}", file=sys.stderr)

                return cache_key, json.loads(cache_path.read_text())
            else:
                if self.cache_debug:
                    rprint(f"[blue]promptrunner {self.name}: No cache found[/blue] {cache_key}", file=sys.stderr)

                if _cache_only:
                    raise Exception(f"No cache found for {self.name} {cache_key}")
        
        return cache_key, None
    
    def run(self, _cache_idx=None, _cache_only=False, **inputs):
        # Format prompt
        prompt, extra_kwargs = self.prompt(**inputs)
        
        # Try cache
        cache_key, cached_output = self.try_cache(_cache_idx=_cache_idx, _cache_only=_cache_only, **inputs)
        if cached_output is not None:
            return cached_output
        
        # Run LLM
        with utils.spinner(f"Running {self.name}"):
            response = completion(
                **self.llm_kwargs,
                num_retries     = 3,
                response_format = self.response_format,
                messages        = [
                    {"role": "system", "content": self.system},
                    {"role": "user",   "content": prompt}
                ]
            )
        
        # [BUG] Why would this happen?
        if len(response.choices) == 0:
            rprint(f"[red]promptrunner {self.name}: No response[/red] {prompt}", file=sys.stderr)
            rprint(response, file=sys.stderr)
            raise Exception(f"No response from {self.name}")
        
        output_str = response.choices[0].message.content
        
        # Structured output
        if self.response_format is not None:
            output = self.response_format.model_validate_json(output_str).model_dump()
        else:
            output = {"output_str" : output_str}
        
        # Post-processing
        if self.after is not None:
            output = self.after(**output, **extra_kwargs)
        
        # Validation
        if self.validate is not None:
            try:
                _ = self.validate(**output)
            except Exception as e:
                rprint(f"[red]promptrunner {self.name}: Validation error[/red] {e}", file=sys.stderr)
                rprint(output, file=sys.stderr)
                raise e
        
        # Write cache
        if self.cache_dir is not None:
            cache_path = Path(self.cache_dir) / f"{cache_key}.json"
            cache_path.write_text(json.dumps(output))

        # Logging
        assert isinstance(output, dict)
        if self.do_log and self.LOG_DIR is not None:
            utils.log(self.LOG_DIR, self.name, self.counter, prompt, output_str, output, show_console=self.do_console)
        
        self.counter += 1
        
        return output

    async def arun(self, _cache_idx=None, _cache_only=False, **inputs):
        # Format prompt
        prompt, extra_kwargs = self.prompt(**inputs)
        
        # Try cache
        cache_key, cached_output = self.try_cache(_cache_idx=_cache_idx, _cache_only=_cache_only, **inputs)
        if cached_output is not None:
            return cached_output
        
        # Run LLM
        # vvvvvv ONLY DIFFERENCE FROM SYNC VERSION vvvvvv
        response = await acompletion(
            **self.llm_kwargs,
            num_retries     = 3,
            response_format = self.response_format,
            messages        = [
                {"role": "system", "content": self.system},
                {"role": "user",   "content": prompt}
            ]
        )
        # ^^^^^^ ONLY DIFFERENCE FROM SYNC VERSION ^^^^^^
        
        # [BUG] Why would this happen?
        if len(response.choices) == 0:
            rprint(f"[red]promptrunner {self.name}: No response[/red] {prompt}", file=sys.stderr)
            rprint(response, file=sys.stderr)
            raise Exception(f"No response from {self.name}")
        
        output_str = response.choices[0].message.content
        
        # Structured output
        if self.response_format is not None:
            output = self.response_format.model_validate_json(output_str).model_dump()
        else:
            output = {"output_str" : output_str}
        
        # Post-processing
        if self.after is not None:
            output = self.after(**output, **extra_kwargs)
        
        # Validation
        if self.validate is not None:
            try:
                _ = self.validate(**output)
            except Exception as e:
                rprint(f"[red]promptrunner {self.name}: Validation error[/red] {e}", file=sys.stderr)
                rprint(output, file=sys.stderr)
                raise e
        
        # Write cache
        if self.cache_dir is not None:
            cache_path = Path(self.cache_dir) / f"{cache_key}.json"
            cache_path.write_text(json.dumps(output))
        
        # Logging
        assert isinstance(output, dict)
        if self.do_log and self.LOG_DIR is not None:
            utils.log(self.LOG_DIR, self.name, self.counter, prompt, output_str, output, show_console=self.do_console)
        
        self.counter += 1
                
        return output
    
    def larun(self, _cache_idx=None, _cache_only=False, **inputs):
        _, cached_output = self.try_cache(_cache_idx, _cache_only, **inputs)
        if cached_output is not None:
            return Future(value=cached_output)
        else:
            async def _fn():
                return await self.arun(_cache_idx=_cache_idx, _cache_only=_cache_only, **inputs)

            return Future(fn=_fn)

