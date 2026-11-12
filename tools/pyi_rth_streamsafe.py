# PyInstaller start-up hook: runs in every process of the frozen exe before its main script, also in the analyzer's
# worker process. It only applies the runtime choice the parent process made (STREAMSAFE_RUNTIME, see
# streamsafe/app/runtime_select.py). Any problem here leaves the bundled runtime in use.
try:
    from streamsafe.app import runtime_select

    runtime_select.activate()
except Exception:  # noqa: S110 - nothing may stop the program from starting; entry.main reports the reason later
    pass
