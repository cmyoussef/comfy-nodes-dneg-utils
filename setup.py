"""Legacy setup.py — kept for compatibility. Source of truth is pyproject.toml."""

from setuptools import setup, find_namespace_packages

setup(
    name="comfy-nodes-dneg-utils",
    version="0.1.0",
    description="DNEG — shared utility ComfyUI custom nodes",
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src", include=["custom_nodes*"]),
    python_requires=">=3.10",
    install_requires=[],
    include_package_data=True,
    package_data={"custom_nodes.dneg_utils_nodes": ["web/*.js", "web/*.css"]},
    entry_points={
        "comfyui.custom_nodes": [
            "dneg_utils_nodes=custom_nodes.dneg_utils_nodes",
        ],
    },
)
