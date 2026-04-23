"""LOB plugin package — importing this package registers all built-in plugins.

Each plugin module calls register() at import time. The imports below
ensure all three builtins are registered whenever accord_ai.lobs is used.
"""
from accord_ai.lobs import commercial_auto, general_liability, workers_comp  # noqa: F401
