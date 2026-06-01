"""Thin progress wrapper around ComfyAPISync."""

_api_sync = None


def _get_api():
    global _api_sync
    if _api_sync is None:
        try:
            from comfy_api.latest import ComfyAPISync
            _api_sync = ComfyAPISync()
        except Exception:
            pass
    return _api_sync


def report_progress(value: int, max_value: int) -> None:
    """Report numeric progress. Silently no-ops if the API is unavailable."""
    api = _get_api()
    if api is None:
        return
    try:
        api.execution.set_progress(value=value, max_value=max_value)
    except Exception:
        pass
