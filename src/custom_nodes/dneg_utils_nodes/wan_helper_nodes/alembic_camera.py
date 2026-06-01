# src\comfyui_remote\custom_nodes\wan_helper_nodes\alembic_camera.py
"""
Alembic camera utilities for extracting camera intrinsics and extrinsics
from .abc files, producing cam_params arrays compatible with the Wan
camera-conditioning pipeline (comfy_extras.nodes_camera_trajectory).

Key concerns handled here, in priority order:

1. Axis convention. Maya/Alembic cameras are right-handed with +Y up and
   look down -Z (the standard DCC / OpenGL convention). Wan was trained on
   CameraCtrl-style data where the local camera basis is +X right, +Y down,
   +Z forward (OpenCV). We convert by post-multiplying c2w with
   diag(1, -1, -1, 1), which flips the camera's local Y and Z axes without
   touching world orientation. `get_relative_pose` downstream then re-anchors
   frame 0 to identity, so world orientation is irrelevant.

2. Translation scale. Alembic units are arbitrary (usually cm in Maya). The
   Wan presets use `base_T_norm = 1.5`, meaning over 81 frames the camera
   travels ~1.5 units. If we hand Wan a 200-unit translation it either
   ignores the conditioning or produces artefacts. We compute relative
   translations (camera position deltas from frame 0), find the max
   magnitude, and scale the whole trajectory so that max equals a target
   range. User can also pass an explicit multiplier to override.

3. Intrinsics. Physical `fx = focal_mm / aperture_mm` values are fine in
   principle, but `process_pose_params` re-scales fx/fy against a hardcoded
   1280x720 reference which assumes CameraCtrl-style normalized intrinsics.
   We default to the preset value (0.5) which corresponds to ~90deg FOV and
   is what Wan saw most during training. Overrides are available for
   advanced use.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

# Import VFX Alembic via alembicAbc wrapper (avoids clash with SQLAlchemy's alembic)
try:
    from alembicAbc import alembic as _abc
    Abc = _abc.Abc
    AbcGeom = _abc.AbcGeom
    HAS_ALEMBIC = True
except ImportError:
    HAS_ALEMBIC = False


AXIS_CONVENTIONS = (
    "maya_to_opencv",   # flip local Y and Z of c2w (default, recommended for Maya .abc)
    "flip_y",           # flip local Y only
    "flip_z",           # flip local Z only
    "swap_yz",          # swap Y and Z columns (for Blender-style Z-up)
    "recammaster",      # reproduces Kijai ReCamMaster: c2w[:,[1,2,0,3]] then flip Y
    "passthrough",      # no conversion (debug / already-converted data)
)


def require_alembic():
    """Raise an error if VFX Alembic bindings are not available."""
    if not HAS_ALEMBIC:
        raise ImportError(
            "VFX Alembic Python bindings (alembicAbc) are required. "
            "Install the alembicAbc package that provides alembic.so."
        )


@dataclass
class CameraFrame:
    """Camera data for a single frame, before any convention conversion."""
    fx: float
    fy: float
    cx: float
    cy: float
    c2w: np.ndarray  # 4x4 camera-to-world matrix (Maya convention)
    focal_length_mm: float
    h_aperture_cm: float
    v_aperture_cm: float


@dataclass
class TrajectoryStats:
    """Diagnostics about the extracted trajectory, useful for sanity checks."""
    num_frames: int
    raw_translation_range: float    # max |T_i - T_0| in source units
    scaled_translation_range: float # max |T_i - T_0| after scaling
    max_rotation_delta_deg: float   # max angle between frame 0 rotation and frame i
    translation_scale_factor: float
    convention: str
    fx: float
    fy: float
    cx: float
    cy: float
    source_focal_mm: float = 0.0
    source_aperture_cm: Tuple[float, float] = (0.0, 0.0)
    warnings: List[str] = field(default_factory=list)

    def to_info_string(self, camera_path: str, start_frame: int) -> str:
        lines = [
            f"Camera: {camera_path}",
            f"Frames: {self.num_frames} (starting at {start_frame})",
            f"Source focal: {self.source_focal_mm:.2f}mm  aperture: "
            f"{self.source_aperture_cm[0]:.3f} x {self.source_aperture_cm[1]:.3f} cm",
            f"Intrinsics used: fx={self.fx:.4f} fy={self.fy:.4f} "
            f"cx={self.cx:.4f} cy={self.cy:.4f}",
            f"Axis convention: {self.convention}",
            f"Translation: raw range {self.raw_translation_range:.3f} "
            f"-> scaled {self.scaled_translation_range:.3f} "
            f"(x{self.translation_scale_factor:.6g})",
            f"Max rotation delta: {self.max_rotation_delta_deg:.2f} deg",
        ]
        for w in self.warnings:
            lines.append(f"WARNING: {w}")
        return "\n".join(lines)


def _get_xform_matrix(xform_schema, time_sample) -> np.ndarray:
    """Extract the 4x4 transformation matrix from an Alembic XformSchema."""
    sample = xform_schema.getValue(time_sample)
    mat = sample.getMatrix()
    result = np.zeros((4, 4), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            result[i, j] = mat[i][j]
    return result


def _compute_world_matrix(obj, time_sample) -> np.ndarray:
    """
    Compute the world transformation matrix by traversing up the hierarchy.
    Accumulates all Xform matrices from the object up to the root.

    Alembic stores row-vector convention matrices (v * M). To compose parents
    correctly we multiply in the order: leaf @ parent @ grandparent @ ...
    which corresponds to reversed(matrices) when matrices are collected from
    leaf upward.
    """
    matrices = []
    current = obj

    while current is not None and current.valid():
        header = current.getHeader()
        if AbcGeom.IXform.matches(header):
            xform = AbcGeom.IXform(current.getParent(), current.getName())
            schema = xform.getSchema()
            if schema.valid():
                mat = _get_xform_matrix(schema, time_sample)
                matrices.append(mat)

        parent = current.getParent()
        if parent is None or not parent.valid() or parent.getName() == "":
            break
        current = parent

    world_matrix = np.eye(4, dtype=np.float64)
    for mat in reversed(matrices):
        world_matrix = world_matrix @ mat

    return world_matrix


def _find_cameras(obj, target_name: str = "", results: Optional[List] = None, path: str = "") -> List[Tuple[str, object]]:
    """
    Recursively search the Alembic hierarchy for ICamera objects.

    Returns list of (full_path, camera_IObject) tuples.
    If target_name is non-empty, only return cameras whose name matches (case-insensitive).
    """
    if results is None:
        results = []

    name = obj.getName()
    full_path = f"{path}/{name}" if path else name

    header = obj.getHeader()
    if AbcGeom.ICamera.matches(header):
        if not target_name or target_name.lower() in name.lower():
            results.append((full_path, obj))

    for i in range(obj.getNumChildren()):
        _find_cameras(obj.getChild(i), target_name, results, full_path)

    return results


def list_cameras_in_abc(abc_path: str) -> List[str]:
    """List all camera objects found in an Alembic file."""
    require_alembic()
    archive = Abc.IArchive(abc_path)
    top = archive.getTop()
    cameras = _find_cameras(top)
    return [path for path, _ in cameras]


# ---------------------------------------------------------------------------
# Convention conversion
# ---------------------------------------------------------------------------

def _convert_c2w(c2w: np.ndarray, convention: str) -> np.ndarray:
    """
    Convert a 4x4 c2w matrix from a source convention to the Wan/OpenCV
    convention (+X right, +Y down, +Z forward).

    We transform the camera's LOCAL axes only (post-multiplication), not
    world. The downstream `get_relative_pose` pins frame 0 to identity, so
    world orientation is irrelevant - only the change in camera basis
    matters.
    """
    if convention == "passthrough":
        return c2w.copy()

    if convention == "maya_to_opencv":
        # Maya/OpenGL camera: +X right, +Y up, -Z forward
        # OpenCV camera:      +X right, +Y down, +Z forward
        flip = np.diag([1.0, -1.0, -1.0, 1.0])
        return c2w @ flip

    if convention == "flip_y":
        return c2w @ np.diag([1.0, -1.0, 1.0, 1.0])

    if convention == "flip_z":
        return c2w @ np.diag([1.0, 1.0, -1.0, 1.0])

    if convention == "swap_yz":
        # Swap local Y and Z (Blender Z-up style)
        m = np.eye(4)
        m[1, 1] = 0.0
        m[1, 2] = 1.0
        m[2, 2] = 0.0
        m[2, 1] = 1.0
        return c2w @ m

    if convention == "recammaster":
        # Reproduces Kijai's ReCamMaster column-swap: X->Z, Y->X, Z->Y, then flip Y
        out = c2w[:, [1, 2, 0, 3]].copy()
        out[:3, 1] *= -1.0
        return out

    raise ValueError(
        f"Unknown axis_convention '{convention}'. "
        f"Valid options: {AXIS_CONVENTIONS}"
    )


def _rotation_angle_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """Angle between two 3x3 rotation matrices, in degrees."""
    R = R1.T @ R2
    trace = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


def _apply_translation_scaling(
    c2ws: np.ndarray,
    translation_scale: float,
    target_motion_range: float,
) -> Tuple[np.ndarray, float, float, float]:
    """
    Scale the translation columns of c2w matrices.

    - If translation_scale > 0: multiply all translations by this factor
      (user override).
    - Else (auto): measure max |T_i - T_0|, compute a scale factor so that
      max becomes target_motion_range.

    Returns: (scaled_c2ws, raw_range, scaled_range, applied_scale_factor)
    """
    positions = c2ws[:, :3, 3]              # (N, 3)
    deltas = positions - positions[0:1]     # (N, 3) relative to frame 0
    magnitudes = np.linalg.norm(deltas, axis=1)
    raw_range = float(magnitudes.max()) if len(magnitudes) > 0 else 0.0

    if translation_scale > 0.0:
        scale = float(translation_scale)
    else:
        if raw_range < 1e-8:
            scale = 1.0  # static camera - nothing to normalize
        else:
            scale = float(target_motion_range) / raw_range

    out = c2ws.copy()
    out[:, :3, 3] *= scale
    scaled_range = raw_range * scale
    return out, raw_range, scaled_range, scale


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_camera_data(
    abc_path: str,
    camera_name: str = "",
    length: int = 81,
    start_frame: int = 0,
    fps: float = 24.0,
    axis_convention: str = "maya_to_opencv",
    translation_scale: float = 0.0,
    target_motion_range: float = 1.5,
    use_preset_intrinsics: bool = True,
    fx_override: float = 0.0,
    fy_override: float = 0.0,
    cx_override: float = 0.0,
    cy_override: float = 0.0,
) -> Tuple[np.ndarray, TrajectoryStats, str]:
    """
    Extract camera data from an Alembic file and build cam_params compatible
    with comfy_extras.nodes_camera_trajectory.process_pose_params.

    Row layout (23 floats per frame):
        [0, fx, fy, cx, cy, 0, 0, c2w_flat_16]

    Args:
        abc_path: Path to the .abc file.
        camera_name: Substring match (case-insensitive); empty = first found.
        length: Number of output frames. For Wan, `(length - 1) % 4 == 0`
            is expected (81, 85, 89, ...).
        start_frame: First frame to read from the Alembic timeline.
        fps: Frames per second for Alembic time sampling.
        axis_convention: One of AXIS_CONVENTIONS. Default "maya_to_opencv".
        translation_scale: Explicit multiplier for all translations. If 0,
            auto-scale so max |T_i - T_0| equals `target_motion_range`.
        target_motion_range: Target max translation magnitude after auto-
            scaling. Wan presets use 1.5 as the base T norm.
        use_preset_intrinsics: If True, ignore the Alembic lens and use
            [fx=0.5, fy=0.5, cx=0.5, cy=0.5] (what the Wan presets use).
            Set False to derive from the Alembic focal/aperture values.
        fx/fy/cx/cy_override: If > 0, these override whatever
            `use_preset_intrinsics` would have produced.

    Returns:
        (cam_params, stats, camera_path)
            cam_params: np.ndarray of shape (length, 23), float32.
            stats:      TrajectoryStats for diagnostics / info display.
            camera_path: Full path of the chosen camera in the .abc hierarchy.
    """
    require_alembic()

    if axis_convention not in AXIS_CONVENTIONS:
        raise ValueError(
            f"axis_convention must be one of {AXIS_CONVENTIONS}, "
            f"got '{axis_convention}'"
        )

    archive = Abc.IArchive(abc_path)
    top = archive.getTop()

    cameras = _find_cameras(top, camera_name)
    if not cameras:
        available = list_cameras_in_abc(abc_path)
        raise ValueError(
            f"No camera found in '{abc_path}'"
            + (f" matching '{camera_name}'" if camera_name else "")
            + f". Available cameras: {available if available else 'none'}"
        )

    if camera_name:
        exact = [(p, o) for p, o in cameras if o.getName().lower() == camera_name.lower()]
        if exact:
            cameras = exact

    cam_path, cam_obj = cameras[0]
    camera = AbcGeom.ICamera(cam_obj.getParent(), cam_obj.getName())
    cam_schema = camera.getSchema()
    num_samples = cam_schema.getNumSamples()

    if num_samples == 0:
        raise ValueError(f"Camera '{cam_path}' has no animation samples")

    warnings: List[str] = []

    # Sample all frames first, then convert + scale in vectorized form.
    raw_c2ws = np.empty((length, 4, 4), dtype=np.float64)
    src_fx = src_fy = src_cx = src_cy = 0.5
    src_focal_mm = 0.0
    src_aperture_cm = (0.0, 0.0)

    for i in range(length):
        frame = start_frame + i
        time = frame / fps
        tss = Abc.ISampleSelector(time)

        cam_sample = cam_schema.getValue(tss)
        focal_mm = cam_sample.getFocalLength()
        h_aperture_cm = cam_sample.getHorizontalAperture()
        v_aperture_cm = cam_sample.getVerticalAperture()
        h_offset_cm = cam_sample.getHorizontalFilmOffset()
        v_offset_cm = cam_sample.getVerticalFilmOffset()

        if i == 0:
            src_focal_mm = focal_mm
            src_aperture_cm = (h_aperture_cm, v_aperture_cm)

            h_aperture_mm = h_aperture_cm * 10.0
            v_aperture_mm = v_aperture_cm * 10.0
            if h_aperture_mm > 0:
                src_fx = focal_mm / h_aperture_mm
            if v_aperture_mm > 0:
                src_fy = focal_mm / v_aperture_mm
            if h_aperture_cm > 0:
                src_cx = 0.5 + (h_offset_cm / h_aperture_cm)
            if v_aperture_cm > 0:
                src_cy = 0.5 + (v_offset_cm / v_aperture_cm)

        raw_c2ws[i] = _compute_world_matrix(cam_obj, tss)

    # --- Choose intrinsics ---
    if use_preset_intrinsics:
        fx, fy, cx, cy = 0.5, 0.5, 0.5, 0.5
    else:
        fx, fy, cx, cy = src_fx, src_fy, src_cx, src_cy

    if fx_override > 0: fx = fx_override
    if fy_override > 0: fy = fy_override
    if cx_override > 0: cx = cx_override
    if cy_override > 0: cy = cy_override

    if not (0.05 <= fx <= 5.0):
        warnings.append(
            f"fx={fx:.3f} is outside the typical training range [0.1, 2]; "
            f"output may be unstable."
        )

    # --- Apply axis convention ---
    converted_c2ws = np.stack(
        [_convert_c2w(raw_c2ws[i], axis_convention) for i in range(length)],
        axis=0,
    )

    # --- Scale translations ---
    scaled_c2ws, raw_range, scaled_range, scale_factor = _apply_translation_scaling(
        converted_c2ws, translation_scale, target_motion_range
    )

    if raw_range < 1e-6 and translation_scale == 0.0:
        warnings.append(
            "Camera is nearly static (no translation). "
            "If you expected movement, check start_frame / length / fps."
        )
    elif scaled_range > 10.0:
        warnings.append(
            f"Scaled translation range {scaled_range:.2f} is large; "
            f"Wan may ignore or distort the camera conditioning."
        )

    # --- Rotation diagnostics ---
    R0 = scaled_c2ws[0, :3, :3]
    max_rot_deg = 0.0
    for i in range(1, length):
        max_rot_deg = max(max_rot_deg, _rotation_angle_deg(R0, scaled_c2ws[i, :3, :3]))

    # --- Build cam_params rows ---
    cam_params = np.empty((length, 23), dtype=np.float32)
    cam_params[:, 0] = 0.0
    cam_params[:, 1] = fx
    cam_params[:, 2] = fy
    cam_params[:, 3] = cx
    cam_params[:, 4] = cy
    cam_params[:, 5] = 0.0
    cam_params[:, 6] = 0.0
    cam_params[:, 7:] = scaled_c2ws.reshape(length, 16).astype(np.float32)

    stats = TrajectoryStats(
        num_frames=length,
        raw_translation_range=raw_range,
        scaled_translation_range=scaled_range,
        max_rotation_delta_deg=max_rot_deg,
        translation_scale_factor=scale_factor,
        convention=axis_convention,
        fx=fx, fy=fy, cx=cx, cy=cy,
        source_focal_mm=src_focal_mm,
        source_aperture_cm=src_aperture_cm,
        warnings=warnings,
    )

    return cam_params, stats, cam_path
