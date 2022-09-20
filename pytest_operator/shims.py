import builtins
import inspect


if not hasattr(builtins, "breakpoint"):
    # Shim breakpoint() builtin from PEP-0553 prior to 3.7
    def _breakpoint(*_, **__):
        import ipdb as ipdb

        current_frame = inspect.currentframe()
        if current_frame:
            ipdb.set_trace(current_frame.f_back)

    builtins.breakpoint = _breakpoint
