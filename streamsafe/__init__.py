"""StreamSafe: real-time covering of risky characters in game streams, before viewers see them.

Pipeline: source -> (analyzer process: detect -> eye -> track/risk -> pose -> garment) ->
output gate (delay buffer, fail closed) -> sink.  Entry points: ``streamsafe.cli`` and
``streamsafe.runtime.runner.run``.
"""

__version__ = "3.0.0"
