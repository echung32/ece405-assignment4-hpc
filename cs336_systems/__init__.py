import importlib.metadata

__version__ = importlib.metadata.version("cs336-systems")

__all__ = ["MODEL_SPECS", "RunConfig", "run_benchmark", "__version__"]


def __getattr__(name: str):
	if name in {"MODEL_SPECS", "RunConfig", "run_benchmark"}:
		from .section1 import benchmarking_script

		return getattr(benchmarking_script, name)
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")