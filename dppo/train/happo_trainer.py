"""Public HAPPO entry point: official marlbenchmark/on-policy implementation."""

try:
    from .official_happo_adapter import HAPPOConfig, HAPPOTrainer
except ImportError:  # pragma: no cover
    from official_happo_adapter import HAPPOConfig, HAPPOTrainer

__all__ = ["HAPPOConfig", "HAPPOTrainer"]

