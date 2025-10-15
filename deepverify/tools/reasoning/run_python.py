import io
import traceback
from contextlib import redirect_stdout, redirect_stderr

def run_python(code: str) -> str:
    """
    Executes a string of Python code and returns its output.
    This is not sandboxed and executes within the main process.
    WARNING: This is insecure and should not be used in a production environment.
    """
    output = io.StringIO()
    error = io.StringIO()

    try:
        with redirect_stdout(output), redirect_stderr(error):
            # Using an empty dict for globals provides a clean namespace
            # but does NOT sandbox the code.
            exec(code, {})
        
        stdout = output.getvalue()
        stderr = error.getvalue()

        result = ""
        if stdout:
            result += f"--- stdout ---\n{stdout}\n"
        if stderr:
            result += f"--- stderr ---\n{stderr}\n"
        if not stdout and not stderr:
            result = "Code executed successfully with no output."
        
        return result

    except Exception:
        return f"--- Exception ---\n{traceback.format_exc()}"
