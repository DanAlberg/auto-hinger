import os

def _log(message: str) -> None:
    # Always-on UI/debug logging for this rework (real-time).
    print(message, flush=True)


def _is_run_json_enabled() -> bool:
    return os.getenv("HINGE_SHOW_RUN_JSON", "0") == "1"


