import builtins
import inspect


if not hasattr(builtins, "breakpoint"):
    # Shim breakpoint() builtin from PEP-0553 prior to 3.7
    def _breakpoint():
        import ipdb as ipdb

        ipdb.set_trace(inspect.currentframe().f_back)

    builtins.breakpoint = _breakpoint
