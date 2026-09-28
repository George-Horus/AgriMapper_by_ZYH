"""Stage 2 model-training package with lazy public entry points."""

__all__ = ["Training_main", "Cross_validation_main"]


def __getattr__(name):
    if name in __all__:
        from .model_train_main import Cross_validation_main, Training_main
        return {"Training_main": Training_main, "Cross_validation_main": Cross_validation_main}[name]
    raise AttributeError(name)
