"""Engine-native adapters for each evaluation domain.

Each adapter pair (env wrapper + DomainAdapter) is self-contained and
loaded directly by the corresponding bench harness in
``src/object_centric_agent/compat/bench_*.py``. We do not eagerly
import them here because the bench harnesses use
``importlib.util.spec_from_file_location`` to dodge the broken
``src/__init__.py`` import chain.

Adapters:
  - arcengine_adapter:        ARC-AGI-3   (environment_files/<game>/<hash>/)
  - baba_adapter:             Baba Is You (nacloos/baba-is-ai)
  - minigrid_engine_adapter:  MiniGrid    (gymnasium/minigrid)
  - minihack_engine_adapter:  MiniHack    (gymnasium/nle/minihack)
  - ocatari_adapter:          OC-Atari    (gymnasium/ale-py/ocatari)
"""
