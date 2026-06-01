from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

from comfy_api.latest import ComfyExtension, io


_VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".m4v",
}
_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}
_TXT_EXTENSIONS = {
    ".txt",
}


def _normalize_extensions(file_type: str) -> set[str] | None:
    """Return a set of allowed suffixes for a given logical file type.

    Args:
        file_type: Logical file type selected in the UI.

    Returns:
        A set of lowercase suffixes including the leading dot,
        or None when all files should be accepted.
    """
    if file_type == "video":
        return _VIDEO_EXTENSIONS
    if file_type == "image":
        return _IMAGE_EXTENSIONS
    if file_type == "txt":
        return _TXT_EXTENSIONS
    if file_type == "any":
        return None

    raise ValueError(f"Unsupported file_type: {file_type}")


def _iter_files(
    directory: Path,
    recursive: bool,
) -> Iterable[Path]:
    """Yield files from directory.

    Args:
        directory: Root directory to scan.
        recursive: Whether to scan recursively.

    Yields:
        File paths only.
    """
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    for path in iterator:
        if path.is_file():
            yield path


class DN_ListFiles(io.ComfyNode):
    """List files from a directory and return them as a path list."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DN_ListFiles",
            display_name="DN List Files",
            category="DN/helpers",
            description=(
                "Scan a directory and return matching file paths as a list. "
                "Useful for batch and dataset workflows, for example: "
                "list video clips -> loop over each file -> run captioning -> save text outputs."
            ),
            inputs=[
                io.String.Input(
                    "directory_path",
                    default="",
                    multiline=False,
                    tooltip=(
                        "Directory to scan for files. Accepts an absolute path or a path "
                        "relative to the ComfyUI server working directory. "
                        "Example: /data/video_clips or ./datasets/clips. "
                        "Provide a directory path, not a file path."
                    ),
                ),
                io.Combo.Input(
                    "file_type",
                    options=["video", "image", "txt", "any"],
                    default="video",
                    tooltip=(
                        "Filter files by a predefined extension group. "
                        "Use 'video' for captioning pipelines, 'image' for image datasets, "
                        "'txt' for text files, or 'any' to include all file types."
                    )
                ),
                io.Boolean.Input(
                    "recursive",
                    default=False,
                    tooltip=(
                        "If enabled, also scan all subdirectories. "
                        "Useful when files are organized in nested folders."
                    ),
                ),
                io.Boolean.Input(
                    "include_hidden",
                    default=False,
                    tooltip="If disabled, files and folders starting with '.' are skipped.",
                ),
                io.Combo.Input(
                    "sort_mode",
                    options=["name", "mtime", "none"],
                    default="name",
                    tooltip=(
                        "Sort the resulting file list by filename, modification time, "
                        "or keep filesystem order."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="file_paths",
                    is_output_list=True,
                ),
                io.Int.Output(
                    display_name="count",
                ),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        directory_path: str,
        **kwargs,
    ) -> bool | str:
        """Validate node inputs before execution."""
        if not directory_path or not directory_path.strip():
            return "directory_path must not be empty."

        directory = Path(directory_path).expanduser()
        if not directory.exists():
            return f"Directory does not exist: {directory}"
        if not directory.is_dir():
            return f"Path is not a directory: {directory}"

        return True

    @classmethod
    def fingerprint_inputs(
        cls,
        directory_path: str,
        file_type: str,
        recursive: bool,
        include_hidden: bool,
        sort_mode: str,
    ) -> str:
        """Build a fingerprint so the node reruns when directory contents change."""
        directory = Path(directory_path).expanduser()

        if not directory.exists() or not directory.is_dir():
            return f"invalid:{directory_path}"

        allowed_extensions = _normalize_extensions(file_type)
        matched_paths: list[Path] = []

        for path in _iter_files(directory=directory, recursive=recursive):
            relative_parts = path.relative_to(directory).parts if path != directory else ()
            if not include_hidden:
                if any(part.startswith(".") for part in relative_parts):
                    continue
                if path.name.startswith("."):
                    continue

            if allowed_extensions is not None:
                if path.suffix.lower() not in allowed_extensions:
                    continue

            matched_paths.append(path)

        matched_paths.sort(key=lambda p: str(p).lower())

        hasher = hashlib.sha256()
        hasher.update(str(directory.resolve()).encode("utf-8"))
        hasher.update(file_type.encode("utf-8"))
        hasher.update(str(recursive).encode("utf-8"))
        hasher.update(str(include_hidden).encode("utf-8"))
        hasher.update(sort_mode.encode("utf-8"))

        for path in matched_paths:
            stat = path.stat()
            hasher.update(str(path.relative_to(directory)).encode("utf-8"))
            hasher.update(str(stat.st_size).encode("utf-8"))
            hasher.update(str(stat.st_mtime_ns).encode("utf-8"))

        return hasher.hexdigest()

    @classmethod
    def execute(
        cls,
        directory_path: str,
        file_type: str,
        recursive: bool,
        include_hidden: bool,
        sort_mode: str,
    ) -> io.NodeOutput:
        """Execute the directory scan.

        Args:
            directory_path: Directory to scan.
            file_type: Logical file type filter.
            recursive: Whether to include subdirectories.
            include_hidden: Whether to include dotfiles / dotdirs.
            sort_mode: Sorting mode.

        Returns:
            NodeOutput with:
                1. file_paths: list[str]
                2. count: int
        """
        directory = Path(directory_path).expanduser().resolve()
        allowed_extensions = _normalize_extensions(file_type)

        matched_paths: list[Path] = []

        for path in _iter_files(directory=directory, recursive=recursive):
            relative_parts = path.relative_to(directory).parts if path != directory else ()
            if not include_hidden:
                if any(part.startswith(".") for part in relative_parts):
                    continue
                if path.name.startswith("."):
                    continue

            if allowed_extensions is not None:
                if path.suffix.lower() not in allowed_extensions:
                    continue

            matched_paths.append(path)

        if sort_mode == "name":
            matched_paths.sort(key=lambda p: str(p).lower())
        elif sort_mode == "mtime":
            matched_paths.sort(key=lambda p: p.stat().st_mtime)
        elif sort_mode == "none":
            pass
        else:
            raise ValueError(f"Unsupported sort_mode: {sort_mode}")

        file_paths = [str(path) for path in matched_paths]
        return io.NodeOutput(file_paths, len(file_paths))

