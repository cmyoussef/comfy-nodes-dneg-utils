import os
from pathlib import Path


def resolve_prompt(prompt_template: str, character_name: str) -> str:
    """Resolves the prompt template by replacing the "{character_name}" token with the actual character name."""
    return prompt_template.format(character_name=character_name)


def resolve_model_name_or_path(model_name_or_path: str) -> str:
    """
    Resolves the model name or path by checking if it's an absolute path, and if not,
    looking for it under COMFY_MODELS_ROOT/models/.

    COMFY_MODELS_ROOT must be set in the environment (e.g. via passthrough_env
    in the enroot configuration of comfy-remote).
    """
    path = Path(model_name_or_path)

    if path.is_absolute():
        return str(path)

    models_root = os.getenv("COMFY_MODELS_ROOT")
    if not models_root:
        raise EnvironmentError(
            "COMFY_MODELS_ROOT is not set. "
            "Set it in the environment or provide an absolute model path."
        )

    return str(Path(models_root) / "models" / model_name_or_path)
