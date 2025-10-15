#!/usr/bin/env python
"""
    promptrunner/utils.py
"""

import litellm
litellm.suppress_debug_info = True

import sys
import json
import asyncio
import numpy as np
from pathlib import Path
from json_repair import repair_json
from tqdm.asyncio import tqdm as atqdm

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich import print as rprint
from time import time, monotonic

# --
# ETL

def json_loads_robust(json_str):
    data = json_str.split('```')[-2]
    if data.startswith('json'):
        data = data[len('json'):]
    
    try:
        return json.loads(data)
    except Exception as e:
        try:
            rprint(f"[yellow]Error parsing JSON ... attempting repair [/yellow]", file=sys.stderr)
            return json.loads(repair_json(data))
        except Exception as e:
            rprint(f"[red]Error parsing JSON ... attempting repair [/red]", file=sys.stderr)
            print(data, file=sys.stderr)
            raise e

# --
# Logging

def spinner(msg, spinner_type="aesthetic"):
    start_time = time()
    console = Console()
    
    def get_spinner_text():
        elapsed = int(time() - start_time)
        return f"{msg} (elapsed: {elapsed}s)"
    
    spinner_obj = Spinner(spinner_type, text=get_spinner_text())
    
    # Create Live display with the spinner
    live = Live(spinner_obj, console=console, refresh_per_second=16)
    
    # Override the get_renderable method to update the elapsed time
    original_get_renderable = live.get_renderable
    def get_renderable():
        spinner_obj.text = get_spinner_text()
        return original_get_renderable()
    
    live.get_renderable = get_renderable
    
    return live

def log(LOG_DIR, name, counter, prompt, output_str, output, show_console=True):
    
    # JSON
    log_path_json = Path(LOG_DIR) / name / f'{name}-{counter:04d}.json'
    log_path_json.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path_json, 'w') as f:
        json.dump({
            "prompt"     : prompt,
            "output_str" : output_str,
            "output"     : output,
        }, f)
    
    # TXT
    if show_console:
        console = Console(record=True)
        
        # Format prompt
        prompt_panel = Panel(prompt, title=f"{name.upper()} - {counter:04d} - INPUT", border_style="blue")
        
        # Format raw output
        output_str_panel = Panel(output_str, title=f"{name.upper()} - {counter:04d} - OUTPUT - RAW", border_style="yellow")
        
        # Format response
        output_text = Pretty(output)
        response_panel = Panel(output_text, title=f"{name.upper()} - {counter:04d} - OUTPUT - FMT", border_style="green")
        
        console.print(prompt_panel, output_str_panel, response_panel)
        
        # Save to file
        log_path_txt  = Path(LOG_DIR) / name / f'{name}-{counter:04d}.txt'
        log_path_txt.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path_txt, 'w') as f:
            f.write(console.export_text())


# --
# API calls

class BatchLogger:
    STATE_CHOICES = ['waiting', 'preparing', 'running', 'complete', 'error']
    def __init__(self):
        self.status     = {}
        self.console    = Console()
        self.live       = None
        self.start_time = time()
    
    def update(self, qid, state):
        assert state in self.STATE_CHOICES, f"Invalid state: {state}"
        self.status[qid] = state

        if not self.live:
            self.live = Live(self._generate_table(), console=self.console, refresh_per_second=4, auto_refresh=True)
            self.live.start()
        else:
            self.live.update(self._generate_table())

        
    def _generate_table(self):
        cnts = {state: 0 for state in self.STATE_CHOICES}
        
        for state in self.status.values():
            cnts[state] += 1
        
        # Create a table with all counts in a single row
        table = Table(expand=True)
        
        # Add columns for each state
        table.add_column("elapsed", justify="center")
        for state in self.STATE_CHOICES:
            table.add_column(f"{state}", justify="center")
        
        # # Add the counts as a single row
        table.add_row(f"{time() - self.start_time:0.3f}s", *[f"{cnts[state]}" for state in self.STATE_CHOICES], style="white")

        return table

    def __del__(self):
        if self.live:
            self.live.stop()

class RateLimiter:
    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self.submission_times = []
        self.lock = asyncio.Lock()
    
    async def __aenter__(self):
        async with self.lock:
            now = monotonic()
            
            # Filter out submission times older than the period
            self.submission_times = [t for t in self.submission_times if now - t < self.period]
            
            if len(self.submission_times) >= self.max_calls:
                oldest       = min(self.submission_times)
                expire_time  = oldest + self.period
                sleep_time   = expire_time - now
                # print(f"Submission limit reached. Waiting {sleep_time:.2f} seconds before submitting the next request.", file=sys.stderr)
                await asyncio.sleep(sleep_time)
                
                # After sleeping, recalculate the submission times list
                now = monotonic()
                self.submission_times = [t for t in self.submission_times if now - t < self.period]
            
            # Add the current time to our submission times
            self.submission_times.append(now)
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


async def arun_batch(futures, max_calls=9999, period=60, delay=0, n_retries=3):
    logger       = BatchLogger()
    rate_limiter = RateLimiter(max_calls=max_calls, period=period)
    
    async def _process_prompt(qid, future):
        if future.is_done:
            logger.update(qid, 'complete')
            return qid, future.value, None

        logger.update(qid, 'waiting')

        await asyncio.sleep(np.random.exponential(delay))
        async with rate_limiter:
            logger.update(qid, 'preparing')
            
            try:
                await asyncio.sleep(np.random.exponential(delay))
                
                logger.update(qid, 'running')
                result = await future()
                logger.update(qid, 'complete')
                
                return qid, result, None
            except Exception as e:
                # rprint(f"[red]Error processing prompt {qid}[/red]: {e}", file=sys.stderr)
                logger.update(qid, 'error')
                return qid, None, e
    
    tasks = [_process_prompt(qid, future) for qid, future in futures.items()]
    
    results = {}
    retries = {}
    for coro in asyncio.as_completed(tasks):
        qid, result, error = await coro
        if error is None:
            results[qid] = result
        else:
            retries[qid] = futures[qid]
    
    del logger

    if len(retries) > 0:
        rprint(f"[yellow]Retrying {len(retries)} failed prompts[/yellow]", file=sys.stderr)
        results.update(await arun_batch(retries, max_calls=max_calls, period=period, delay=delay, n_retries=n_retries - 1))

    return {qid: results[qid] for qid in futures.keys()}


def run_batch(*args, **kwargs):
    # Create a new event loop each time
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(arun_batch(*args, **kwargs))
    finally:
        loop.close()

