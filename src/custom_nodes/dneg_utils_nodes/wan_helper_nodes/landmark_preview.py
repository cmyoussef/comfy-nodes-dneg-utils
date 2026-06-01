"""
Landmark Preview node for ComfyUI.

Point it at a directory of extracted face PNGs. Each PNG contains both the
image pixels and DFL metadata (landmarks, pose, etc.) in an ``fcWp`` chunk.
The node reads everything it needs from the files.
"""
from __future__ import annotations

import glob
import hashlib
import json
import pickle
import re
import struct
from pathlib import Path

import cv2
import numpy as np
import torch
from comfy_api.latest import io

from ._progress import report_progress

_PNG_HEADER = b"\x89PNG\r\n\x1a\n"


def _read_dfl_dict(filepath: str | Path) -> dict | None:
    """Read the DFL metadata dict from a PNG file."""
    filepath = Path(filepath)
    if not filepath.exists():
        return None
    with open(filepath, "rb") as fh:
        header = fh.read(8)
        if header != _PNG_HEADER:
            return None

        while True:
            raw = fh.read(8)
            if len(raw) < 8:
                break
            length, name = struct.unpack(">I4s", raw)
            data = fh.read(length)
            fh.read(4)  # CRC

            if name == b"fcWp":
                try:
                    return pickle.loads(data)
                except Exception:
                    return None
    return None


_FACE_PARTS = {
    "jaw": list(range(0, 17)),
    "right_brow": list(range(17, 22)),
    "left_brow": list(range(22, 27)),
    "nose_bridge": list(range(27, 31)),
    "nose_tip": list(range(31, 36)),
    "right_eye": list(range(36, 42)) + [36],
    "left_eye": list(range(42, 48)) + [42],
    "outer_lip": list(range(48, 60)) + [48],
    "inner_lip": list(range(60, 68)) + [60],
}

_COLORS_AI = {k: (0, 255, 0) for k in _FACE_PARTS}
_COLORS_BT = {k: (0, 165, 255) for k in _FACE_PARTS}

_COLORS_AI_RAINBOW = {
    "jaw": (255, 0, 0),
    "right_brow": (255, 127, 0),
    "left_brow": (0, 127, 255),
    "nose_bridge": (255, 255, 0),
    "nose_tip": (0, 255, 255),
    "right_eye": (0, 255, 0),
    "left_eye": (0, 200, 0),
    "outer_lip": (255, 0, 255),
    "inner_lip": (200, 0, 200),
}

_COLORS_BT_RAINBOW = {
    "jaw": (200, 50, 50),
    "right_brow": (200, 127, 50),
    "left_brow": (50, 127, 200),
    "nose_bridge": (200, 200, 50),
    "nose_tip": (50, 200, 200),
    "right_eye": (50, 200, 50),
    "left_eye": (50, 160, 50),
    "outer_lip": (200, 50, 200),
    "inner_lip": (160, 50, 160),
}


def _draw_landmarks_68(
    canvas: np.ndarray,
    landmarks: np.ndarray,
    colors: dict[str, tuple[int, int, int]],
    thickness: int = 1,
    radius: int = 2,
    alpha: float = 0.8,
) -> np.ndarray:
    overlay = canvas.copy()
    pts = landmarks.astype(np.int32)

    for part_name, indices in _FACE_PARTS.items():
        color = colors.get(part_name, (0, 255, 0))
        for i in range(len(indices) - 1):
            p1 = tuple(pts[indices[i]])
            p2 = tuple(pts[indices[i + 1]])
            cv2.line(overlay, p1, p2, color, thickness, cv2.LINE_AA)
        for idx in indices:
            cv2.circle(overlay, tuple(pts[idx]), radius, color, -1, cv2.LINE_AA)

    if alpha < 1.0:
        return cv2.addWeighted(canvas, 1.0 - alpha, overlay, alpha, 0)
    return overlay


def _bgr_to_tensor(bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.astype(np.float32) / 255.0).unsqueeze(0)


def _compute_hash(**kwargs) -> str:
    m = hashlib.sha256()
    for k in sorted(kwargs):
        v = kwargs[k]
        if v is None:
            m.update(f"{k}:N".encode())
        else:
            m.update(f"{k}:{v}".encode("utf-8", errors="ignore"))
    return m.hexdigest()


def _files_hash(paths: list[str]) -> str:
    m = hashlib.sha256()
    for p in sorted(paths):
        fp = Path(p)
        if fp.exists():
            s = fp.stat()
            m.update(f"{p}:{s.st_mtime}:{s.st_size}".encode())
    return m.hexdigest()


def _parse_input_to_paths(value: str) -> tuple[list[str], str]:
    """
    Accept:
      - JSON string: '["/p1.png", "/p2.png"]' or {"paths":[...]}
      - JSON manifest file path (.json)
      - Directory path
      - Glob / wildcard / sequence-ish patterns
      - Single file path
    Returns (paths, mode)
    """
    s = (value or "").strip()
    if s == "" or s.lower() in {"none", "null"}:
        return [], "empty"

    # JSON string payload
    try:
        obj = json.loads(s)
        if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
            return obj, "json"
        if isinstance(obj, str):
            return [obj], "json"
        if isinstance(obj, dict):
            for key in ("paths", "files", "items"):
                v = obj.get(key)
                if isinstance(v, list) and all(isinstance(x, str) for x in v):
                    return v, "json"
    except Exception:
        pass

    p = Path(s)

    # JSON manifest file path
    if p.is_file() and p.suffix.lower() == ".json":
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
                return obj, "json_manifest"
            if isinstance(obj, str):
                return [obj], "json_manifest"
            if isinstance(obj, dict):
                for key in ("paths", "files", "items"):
                    v = obj.get(key)
                    if isinstance(v, list) and all(isinstance(x, str) for x in v):
                        return v, "json_manifest"
        except Exception:
            # fall back to plain path handling
            pass

    # Directory
    if p.exists() and p.is_dir():
        return [s], "dir"

    # Sequence/glob-like patterns
    if "%" in s or "#" in s or " [" in s:
        fp = re.sub(r"#+", "*", s)
        fp = re.sub(r"%0\d+d", "*", fp)
        return sorted(glob.glob(fp)), "sequence"

    if any(c in s for c in ["*", "?", "["]):
        return sorted(glob.glob(s)), "glob"

    # Single path
    return [s], "path"


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr", ".bmp", ".tga", ".hdr", ".dpx"}


def _expand_to_image_files(paths: list[str]) -> list[str]:
    """
    Expand mixed inputs (dirs/files/pattern results) into a sorted list of image files.
    If a directory contains a 'faces' subdir, that subdir is preferred.
    """
    out: list[str] = []
    for raw in paths:
        if not raw:
            continue
        p = Path(raw)
        if p.exists() and p.is_dir():
            d = p / "faces" if (p / "faces").is_dir() else p
            out.extend(
                str(f) for f in sorted(d.iterdir())
                if f.is_file() and f.suffix.lower() in _IMAGE_EXTS
            )
            continue
        if p.suffix.lower() in _IMAGE_EXTS:
            out.append(str(p))
    # stable dedupe
    return sorted(dict.fromkeys(out))


LANDMARK_TYPES = ["ai", "bodytrack", "both"]
COLOR_SCHEMES = ["default", "rainbow"]
BACKGROUND_MODES = ["original", "black"]


class WanHelper_LandmarkPreview(io.ComfyNode):
    DESCRIPTION = (
        "Read face PNGs and draw landmarks from embedded metadata.\n\n"
        "Each PNG contains the face image and a DFL metadata chunk with\n"
        "landmark coordinates, pose, etc. Just point faces_dir at them.\n\n"
        "Landmark types:\n"
        "  ai        - AI-detected 68-point landmarks (green)\n"
        "  bodytrack - Body-track projected landmarks (orange)\n"
        "  both      - Overlay both for comparison\n\n"
        "Background:\n"
        "  original  - Draw on the face image\n"
        "  black     - Draw on a black canvas"
    )

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WanHelper_LandmarkPreview",
            display_name="WanHelper: Landmark Preview",
            category="WanHelper/preview",
            description=cls.DESCRIPTION,
            inputs=[
                io.String.Input(
                    "faces_dir",
                    default="",
                    multiline=False,
                    tooltip=(
                        "Directory, single PNG path, glob/sequence pattern, JSON list string,\n"
                        "or JSON manifest file path. If a directory contains '/faces', it is used."
                    ),
                ),
                io.Combo.Input(
                    "landmark_type",
                    options=LANDMARK_TYPES,
                    default="both",
                    tooltip="Which landmarks to draw.",
                ),
                io.Combo.Input(
                    "background",
                    options=BACKGROUND_MODES,
                    default="original",
                    tooltip="Background mode for drawing.",
                ),
                io.Combo.Input(
                    "color_scheme",
                    options=COLOR_SCHEMES,
                    default="default",
                    tooltip="default = green/orange, rainbow = per-region colors.",
                ),
                io.Int.Input(
                    "line_thickness",
                    default=1,
                    min=1,
                    max=5,
                    step=1,
                    tooltip="Thickness of landmark connection lines.",
                ),
                io.Int.Input(
                    "circle_radius",
                    default=2,
                    min=1,
                    max=10,
                    step=1,
                    tooltip="Radius of landmark point circles.",
                ),
                io.Float.Input(
                    "alpha",
                    default=0.8,
                    min=0.1,
                    max=1.0,
                    step=0.1,
                    tooltip="Opacity of landmark overlay.",
                ),
                io.String.Input(
                    "extraction_output",
                    optional=True,
                    force_input=True,
                    tooltip="Face extractor output dir; '/faces' auto-appended if needed.",
                ),
            ],
            outputs=[
                io.Image.Output("images", tooltip="Face images with landmarks drawn [B,H,W,C]."),
                io.String.Output("info", tooltip="Per-face metadata summary."),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, faces_dir: str = "", **kwargs) -> str:
        actual = kwargs.get("extraction_output", faces_dir) or faces_dir
        base_paths, _mode = _parse_input_to_paths(actual)
        paths = _expand_to_image_files(base_paths)
        fh = _files_hash(paths) if paths else "none"
        ph = _compute_hash(
            faces_dir=faces_dir,
            extraction_output=kwargs.get("extraction_output", ""),
            landmark_type=kwargs.get("landmark_type", "both"),
            background=kwargs.get("background", "original"),
            color_scheme=kwargs.get("color_scheme", "default"),
            line_thickness=kwargs.get("line_thickness", 1),
            circle_radius=kwargs.get("circle_radius", 2),
            alpha=kwargs.get("alpha", 0.8),
        )
        return f"{ph}_{fh}"

    @classmethod
    def execute(
        cls,
        faces_dir: str,
        landmark_type: str,
        background: str,
        color_scheme: str,
        line_thickness: int,
        circle_radius: int,
        alpha: float,
        extraction_output: str = "",
    ) -> io.NodeOutput:
        source = extraction_output or faces_dir
        base_paths, source_mode = _parse_input_to_paths(source)
        face_files = [Path(p) for p in _expand_to_image_files(base_paths)]
        if not face_files:
            raise ValueError(
                "No image files resolved from input. "
                "Provide a directory, image path, JSON list, JSON manifest, or glob pattern."
            )

        if color_scheme == "rainbow":
            ai_colors, bt_colors = _COLORS_AI_RAINBOW, _COLORS_BT_RAINBOW
        else:
            ai_colors, bt_colors = _COLORS_AI, _COLORS_BT

        results: list[torch.Tensor] = []
        info_lines = [
            "Landmark Preview",
            "=" * 40,
            f"Source: {source_mode}",
            f"Input: {source}",
            f"Faces: {len(face_files)}",
            f"Landmark type: {landmark_type}  |  Background: {background}",
            "",
        ]

        no_ai = 0
        no_bt = 0
        skipped = 0

        for file_idx, face_file in enumerate(face_files):
            img_bgr = cv2.imread(str(face_file))
            if img_bgr is None:
                skipped += 1
                continue

            h, w = img_bgr.shape[:2]
            dfl = _read_dfl_dict(face_file)

            ai_lm = None
            bt_lm = None
            if dfl is not None:
                raw = dfl.get("landmarks")
                if raw is not None:
                    arr = np.array(raw, dtype=np.float32)
                    if arr.shape == (68, 2):
                        ai_lm = arr

                raw = dfl.get("l_bt_kps68")
                if raw is not None:
                    arr = np.array(raw, dtype=np.float32)
                    if arr.shape == (68, 2):
                        bt_lm = arr

            if ai_lm is None:
                no_ai += 1
            if bt_lm is None:
                no_bt += 1

            if background == "black":
                canvas = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                canvas = img_bgr.copy()

            if landmark_type in ("bodytrack", "both") and bt_lm is not None:
                canvas = _draw_landmarks_68(
                    canvas, bt_lm, bt_colors,
                    thickness=line_thickness, radius=circle_radius, alpha=alpha,
                )
            if landmark_type in ("ai", "both") and ai_lm is not None:
                canvas = _draw_landmarks_68(
                    canvas, ai_lm, ai_colors,
                    thickness=line_thickness, radius=circle_radius, alpha=alpha,
                )

            results.append(_bgr_to_tensor(canvas))

            pose = dfl.get("pose") if dfl else None
            pose_str = (
                f"pitch={pose[0]:.1f} yaw={pose[1]:.1f} roll={pose[2]:.1f}"
                if pose and len(pose) >= 3 else "n/a"
            )
            info_lines.append(
                f"  {face_file.name}  "
                f"ai={'yes' if ai_lm is not None else 'no'}  "
                f"bt={'yes' if bt_lm is not None else 'no'}  "
                f"pose=[{pose_str}]"
            )
            report_progress(file_idx + 1, len(face_files))

        if not results:
            raise ValueError("No valid face images could be loaded")

        batch = torch.cat(results, dim=0)
        info_lines.extend([
            "",
            f"Loaded: {len(results)}  |  Skipped: {skipped}  "
            f"|  Missing AI: {no_ai}  |  Missing BT: {no_bt}",
        ])

        return io.NodeOutput(batch, "\n".join(info_lines))
