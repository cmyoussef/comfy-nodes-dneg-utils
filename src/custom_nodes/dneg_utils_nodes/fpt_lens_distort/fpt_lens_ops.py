# src\comfyui_remote\custom_nodes\fpt_lens_distort\fpt_lens_ops.py
"""
FPT Lens Distortion Operations
All math, IO helpers, and core functions for lens distortion/undistortion.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any, Dict

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(frozen=True)
class FPTCameraCalib:
    """
    Camera calibration data.

    @param name: camera name (from metadata.camera or metadata.name)
    @param K: 3x3 intrinsics matrix (float64)
    @param dist: distortion coefficients (float64), length 5 or 8
    @param authored_size: (width, height) resolution calibration was created for
    """
    name: str
    K: np.ndarray
    dist: np.ndarray
    authored_size: Tuple[int, int]


@dataclass(frozen=True)
class FPTSTMapSpec:
    """
    STMap specification.

    @param kind: what this map produces when applied ("undistort" or "distort")
    @param space: coordinate space ("uv_0_1" normalized or "pixel_xy" absolute)
    @param channels: which channels store U/V ("RG" or "BA")
    @param resample: allow resizing stmap to match image size
    """
    kind: str
    space: str
    channels: str
    resample: bool


# =============================================================================
# Validation
# =============================================================================

def require_cv2() -> None:
    """Ensure OpenCV is available."""
    if cv2 is None:
        raise RuntimeError(
            "OpenCV (cv2) is not available. "
            "Install opencv-python in your ComfyUI environment."
        )


# =============================================================================
# File IO
# =============================================================================

def read_calib_file(filepath: str) -> Tuple[str, str]:
    """
    Read calibration file from path.

    @param filepath: path to JSON file
    @return: (text_content, sha256_hex)
    """
    if not filepath or not filepath.strip():
        raise ValueError("No calibration file path provided.")

    filepath = filepath.strip()
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Calibration file not found: {filepath}")

    with open(filepath, "rb") as f:
        raw = f.read()

    sha = hashlib.sha256(raw).hexdigest()
    return raw.decode("utf-8"), sha


# =============================================================================
# JSON Parsing
# =============================================================================

def parse_camera_from_json(json_text: str, camera_name: str) -> FPTCameraCalib:
    """
    Parse camera calibration from JSON.

    @param json_text: JSON string containing list of camera objects
    @param camera_name: name to match against metadata.camera or metadata.name
    @return: parsed camera calibration
    """
    data = json.loads(json_text)
    if not isinstance(data, list):
        raise ValueError("Calibration JSON must be a list of objects.")

    selected = None
    available: List[str] = []

    for item in data:
        if not isinstance(item, dict):
            continue

        md = item.get("metadata") or {}
        # Skip non-camera entries
        if md.get("type") not in ("camera", None) or "fx" not in item:
            continue

        name = md.get("camera") or md.get("name")
        if name is not None:
            available.append(str(name))

        if str(name) == str(camera_name):
            selected = item

    if selected is None:
        raise ValueError(f"Camera '{camera_name}' not found. Available: {available}")

    # Validate distortion model
    dist_model = str(selected.get("distortion_model", "opencv")).lower()
    if dist_model != "opencv":
        raise ValueError(f"Unsupported distortion_model '{dist_model}'. Only 'opencv' supported.")

    # Extract intrinsics
    fx = float(selected["fx"])
    fy = float(selected["fy"])
    cx = float(selected["cx"])
    cy = float(selected["cy"])
    w = int(selected["image_size_x"])
    h = int(selected["image_size_y"])

    # Extract distortion coefficients
    k1 = float(selected.get("k1", 0.0))
    k2 = float(selected.get("k2", 0.0))
    p1 = float(selected.get("p1", 0.0))
    p2 = float(selected.get("p2", 0.0))
    k3 = float(selected.get("k3", 0.0))

    # Optional rational model coefficients (k4, k5, k6)
    k4 = selected.get("k4")
    k5 = selected.get("k5")
    k6 = selected.get("k6")

    if k4 is None and k5 is None and k6 is None:
        dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
    else:
        dist = np.array([
            k1, k2, p1, p2, k3,
            float(k4 or 0.0),
            float(k5 or 0.0),
            float(k6 or 0.0)
        ], dtype=np.float64)

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    return FPTCameraCalib(
        name=str(camera_name),
        K=K,
        dist=dist,
        authored_size=(w, h)
    )


def get_available_cameras(json_text: str) -> List[str]:
    """
    Get list of camera names from calibration JSON.

    @param json_text: JSON string
    @return: list of camera names
    """
    data = json.loads(json_text)
    if not isinstance(data, list):
        return []

    names = []
    for item in data:
        if not isinstance(item, dict):
            continue
        md = item.get("metadata") or {}
        if "fx" not in item:
            continue
        name = md.get("camera") or md.get("name")
        if name:
            names.append(str(name))
    return names


# =============================================================================
# Intrinsics Scaling
# =============================================================================

def scale_intrinsics(
    K: np.ndarray,
    authored_size: Tuple[int, int],
    image_size: Tuple[int, int]
) -> np.ndarray:
    """
    Scale camera intrinsics for different image resolution.

    @param K: 3x3 intrinsics matrix
    @param authored_size: (width, height) original calibration size
    @param image_size: (width, height) target size
    @return: scaled 3x3 intrinsics
    """
    aw, ah = authored_size
    iw, ih = image_size
    sx = float(iw) / float(aw)
    sy = float(ih) / float(ah)

    Ks = K.copy()
    Ks[0, 0] *= sx  # fx
    Ks[1, 1] *= sy  # fy
    Ks[0, 2] *= sx  # cx
    Ks[1, 2] *= sy  # cy
    return Ks


# =============================================================================
# OpenCV Helpers
# =============================================================================

def cv_border_mode(name: str) -> int:
    """Convert border mode name to cv2 constant."""
    require_cv2()
    modes = {
        "replicate": cv2.BORDER_REPLICATE,
        "reflect": cv2.BORDER_REFLECT_101,
        "constant": cv2.BORDER_CONSTANT,
    }
    return modes.get(name, cv2.BORDER_CONSTANT)


def cv_interpolation(name: str) -> int:
    """Convert interpolation name to cv2 constant."""
    require_cv2()
    interps = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    return interps.get(name, cv2.INTER_LINEAR)


# =============================================================================
# Map Building - OpenCV Calibration
# =============================================================================

def _build_distort_maps(
    K: np.ndarray,
    dist: np.ndarray,
    new_K: np.ndarray,
    image_size: Tuple[int, int],
    chunk_rows: int = 128
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build distortion maps (for applying distortion to an undistorted image).

    For cv2.remap: dst[y,x] = src[map_y, map_x]
    Here: dst = distorted output, src = undistorted input

    For each pixel (x,y) in distorted output space:
    - Use undistortPoints to find where it maps in undistorted space
    - That's where we sample from the undistorted source

    @param K: original (distorted) camera matrix
    @param dist: distortion coefficients
    @param new_K: new camera matrix for undistorted image
    @param image_size: (width, height)
    @param chunk_rows: rows per chunk for memory efficiency
    @return: (map1, map2) float32 arrays
    """
    require_cv2()

    w, h = image_size
    map1 = np.empty((h, w), dtype=np.float32)
    map2 = np.empty((h, w), dtype=np.float32)

    xs = np.arange(w, dtype=np.float32)

    for y0 in range(0, h, chunk_rows):
        y1 = min(h, y0 + chunk_rows)
        rows = y1 - y0

        ys = np.arange(y0, y1, dtype=np.float32)
        xv = np.tile(xs[None, :], (rows, 1))
        yv = np.tile(ys[:, None], (1, w))

        # Points in distorted output image coordinates
        pts = np.stack([xv, yv], axis=-1).reshape(-1, 1, 2).astype(np.float32)

        # undistortPoints: distorted pixel coords -> undistorted pixel coords
        # With P=new_K, output is in new_K pixel space (undistorted image)
        und = cv2.undistortPoints(pts, K, dist, R=None, P=new_K)
        und = und.reshape(rows, w, 2)

        map1[y0:y1, :] = und[..., 0]
        map2[y0:y1, :] = und[..., 1]

    return map1.astype(np.float32), map2.astype(np.float32)


def build_opencv_maps(
    cam: FPTCameraCalib,
    image_size: Tuple[int, int],
    balance: float,
    scale_to_input: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """
    Build both undistort and distort remap arrays.

    OpenCV distortion model (for 5 coefficients):
    x'' = x' * (1 + k1*r² + k2*r⁴ + k3*r⁶) + 2*p1*x'*y' + p2*(r² + 2*x'²)
    y'' = y' * (1 + k1*r² + k2*r⁴ + k3*r⁶) + p1*(r² + 2*y'²) + 2*p2*x'*y'

    Where x' = (u - cx)/fx, y' = (v - cy)/fy (normalized coordinates)
    and r² = x'² + y'²

    @param cam: camera calibration
    @param image_size: (width, height) of input image
    @param balance: alpha for getOptimalNewCameraMatrix (0=crop, 1=keep all)
    @param scale_to_input: scale intrinsics if size differs from authored
    @return: (map_ud_1, map_ud_2, map_du_1, map_du_2, roi)
    """
    require_cv2()

    iw, ih = image_size
    K = cam.K.copy()
    dist = cam.dist

    # Scale intrinsics if image size differs from calibration size
    if (iw, ih) != cam.authored_size:
        if not scale_to_input:
            raise ValueError(
                f"Image size {iw}x{ih} != calibration size "
                f"{cam.authored_size[0]}x{cam.authored_size[1]}. "
                "Enable scale_to_input or use matching resolution."
            )
        K = scale_intrinsics(cam.K, cam.authored_size, (iw, ih))

    # Get optimal new camera matrix for undistorted output
    # balance=0: crop to valid pixels, balance=1: keep all with black borders
    new_K, roi = cv2.getOptimalNewCameraMatrix(
        K, dist, (iw, ih), float(balance), (iw, ih)
    )

    # Undistort maps: for each pixel in undistorted output,
    # where to sample from distorted input
    # dst_undistorted[y,x] = src_distorted[map_y, map_x]
    map_ud_1, map_ud_2 = cv2.initUndistortRectifyMap(
        K, dist, None, new_K, (iw, ih), cv2.CV_32FC1
    )

    # Distort maps: for each pixel in distorted output,
    # where to sample from undistorted input
    # dst_distorted[y,x] = src_undistorted[map_y, map_x]
    map_du_1, map_du_2 = _build_distort_maps(K, dist, new_K, (iw, ih))

    return map_ud_1, map_ud_2, map_du_1, map_du_2, roi


# =============================================================================
# Unbulge - Localized Center Shrink for HMC Footage
# =============================================================================

def build_unbulge_maps(
    image_size: Tuple[int, int],
    strength: float,
    center_x: float = 0.5,
    center_y: float = 0.5,
    inner_radius: float = 0.15,
    outer_radius: float = 0.5,
    scale: float = 1.0,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build remap arrays for localized center shrink (unbulge) with scale.

    Uses smooth cosine falloff between inner and outer radius.
    Scale is applied POST (after unbulge): 1. Unbulge  2. Scale

    @param image_size: (width, height)
    @param strength: shrink amount (0.0 to 1.0)
    @param center_x: center X as fraction (0.5 = center)
    @param center_y: center Y as fraction (0.5 = center)
    @param inner_radius: full effect inside this radius (fraction of image)
    @param outer_radius: effect fades to zero at this radius
    @param scale: uniform scale applied after unbulge (>1 = zoom in, <1 = zoom out)
    @param scale_x: horizontal scale multiplier
    @param scale_y: vertical scale multiplier
    @return: (map_x, map_y, weight) float32 arrays
    """
    require_cv2()

    w, h = image_size

    # Create coordinate grids
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    map_x, map_y = np.meshgrid(xs, ys)

    # Center in pixels
    cx = center_x * w
    cy = center_y * h

    # Combined scale factors
    total_scale_x = scale * scale_x
    total_scale_y = scale * scale_y

    # Distance from center (original coords for unbulge calculation)
    dx = map_x - cx
    dy = map_y - cy

    # Radii in pixels
    inner_r = inner_radius * max(w, h)
    outer_r = outer_radius * max(w, h)

    # Distance from center
    dist = np.sqrt(dx**2 + dy**2)

    # Compute weight with smooth falloff
    falloff_range = max(outer_r - inner_r, 1e-6)
    t = np.clip((dist - inner_r) / falloff_range, 0, 1)
    weight = np.cos(t * np.pi / 2) ** 2

    # Unbulge: scale > 1 = sample from further out = shrink center
    unbulge_scale = 1.0 + strength * weight

    # Apply unbulge first, then scale POST (after)
    # Order: 1. Unbulge  2. Scale
    x_src = cx + dx * unbulge_scale / total_scale_x
    y_src = cy + dy * unbulge_scale / total_scale_y

    return x_src.astype(np.float32), y_src.astype(np.float32), weight.astype(np.float32)


def apply_unbulge(
    image: np.ndarray,
    strength: float,
    center_x: float = 0.5,
    center_y: float = 0.5,
    inner_radius: float = 0.15,
    outer_radius: float = 0.5,
    scale: float = 1.0,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    interpolation: str = "lanczos",
    border: str = "replicate",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply localized center shrink with scale controls.
    Scale is applied POST: 1. Unbulge (shrink nose)  2. Scale (zoom to frame)

    @param image: (H, W, C) float32 array
    @param strength: shrink amount (0-1)
    @param center_x: center X (0-1)
    @param center_y: center Y (0-1)
    @param inner_radius: full effect zone (fraction of image)
    @param outer_radius: effect ends here (fraction of image)
    @param scale: uniform scale after unbulge (>1 zoom in, <1 zoom out)
    @param scale_x: horizontal scale multiplier
    @param scale_y: vertical scale multiplier
    @param interpolation: "nearest", "linear", "cubic", "lanczos"
    @param border: "constant", "replicate", "reflect"
    @return: (corrected_image, weight_mask)
    """
    require_cv2()

    h, w = image.shape[:2]

    map_x, map_y, weight = build_unbulge_maps(
        image_size=(w, h),
        strength=strength,
        center_x=center_x,
        center_y=center_y,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        scale=scale,
        scale_x=scale_x,
        scale_y=scale_y,
    )

    interp = cv_interpolation(interpolation)
    border_mode = cv_border_mode(border)

    # Remap
    result = cv2.remap(
        image, map_x, map_y, interp,
        borderMode=border_mode,
        borderValue=0.0
    )

    return result, weight


def generate_stmap_from_calib(
    cam: FPTCameraCalib,
    image_size: Tuple[int, int],
    mode: str,
    balance: float = 0.0,
    scale_to_input: bool = True,
    normalize: bool = True
) -> np.ndarray:
    """
    Generate an STMap image from camera calibration.
    Useful for debugging or exporting to other software.

    @param cam: camera calibration
    @param image_size: (width, height)
    @param mode: "undistort" or "distort"
    @param balance: alpha for getOptimalNewCameraMatrix
    @param scale_to_input: scale intrinsics if needed
    @param normalize: if True, output UV in [0,1], else pixel coordinates
    @return: (H, W, 3) float32 array with R=U, G=V, B=0
    """
    require_cv2()

    iw, ih = image_size

    map_ud_1, map_ud_2, map_du_1, map_du_2, roi = build_opencv_maps(
        cam, image_size, balance, scale_to_input
    )

    if mode == "undistort":
        map_x, map_y = map_ud_1, map_ud_2
    else:
        map_x, map_y = map_du_1, map_du_2

    if normalize:
        # Normalize to [0, 1] range
        u = map_x / float(iw - 1)
        v = map_y / float(ih - 1)
    else:
        u = map_x
        v = map_y

    # Create RGB image with R=U, G=V, B=0
    stmap = np.zeros((ih, iw, 3), dtype=np.float32)
    stmap[..., 0] = u
    stmap[..., 1] = v

    return stmap


# =============================================================================
# Map Building - STMap
# =============================================================================

def _extract_uv_channels(
    stmap: np.ndarray,
    channels: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract U/V channels from STMap.

    @param stmap: (H, W, C) float32 array
    @param channels: "RG" or "BA"
    @return: (u, v) float32 arrays
    """
    if stmap.ndim != 3 or stmap.shape[2] < 2:
        raise ValueError("STMap must have at least 2 channels.")

    if channels == "BA":
        if stmap.shape[2] < 4:
            raise ValueError("STMap channels=BA requires 4-channel input.")
        u = stmap[..., 2]
        v = stmap[..., 3]
    else:  # RG
        u = stmap[..., 0]
        v = stmap[..., 1]

    return u.astype(np.float32), v.astype(np.float32)


def _resize_uv_maps(
    u: np.ndarray,
    v: np.ndarray,
    target_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resize UV maps to target size.

    @param u: (H, W) float32
    @param v: (H, W) float32
    @param target_size: (width, height)
    @return: resized (u, v)
    """
    require_cv2()

    tw, th = target_size
    if u.shape[1] == tw and u.shape[0] == th:
        return u, v

    u_resized = cv2.resize(u, (tw, th), interpolation=cv2.INTER_LINEAR)
    v_resized = cv2.resize(v, (tw, th), interpolation=cv2.INTER_LINEAR)

    return u_resized.astype(np.float32), v_resized.astype(np.float32)


def build_stmap_maps(
    stmap: np.ndarray,
    image_size: Tuple[int, int],
    spec: FPTSTMapSpec
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build cv2.remap maps from STMap.

    @param stmap: (H, W, C) float32 STMap
    @param image_size: (width, height)
    @param spec: STMap specification
    @return: (map1, map2) float32 remap arrays
    """
    require_cv2()

    iw, ih = image_size
    u, v = _extract_uv_channels(stmap, spec.channels)

    # Resize if needed
    if spec.resample:
        u, v = _resize_uv_maps(u, v, (iw, ih))
    else:
        if u.shape[1] != iw or u.shape[0] != ih:
            raise ValueError(
                f"STMap size {u.shape[1]}x{u.shape[0]} != image size {iw}x{ih}. "
                "Enable stmap_resample to allow resizing."
            )

    # Convert to pixel coordinates if needed
    if spec.space == "pixel_xy":
        map1, map2 = u, v
    else:  # uv_0_1
        map1 = u * float(iw - 1)
        map2 = v * float(ih - 1)

    return map1.astype(np.float32), map2.astype(np.float32)


# =============================================================================
# Map Inversion
# =============================================================================

def invert_maps(
    map1: np.ndarray,
    map2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Invert dense remap mapping (best effort).

    @param map1: (H, W) float32 source X coordinates
    @param map2: (H, W) float32 source Y coordinates
    @return: (inv_map1, inv_map2) inverted maps
    """
    require_cv2()

    # Use cv2.invertMaps if available (OpenCV 4.x)
    if hasattr(cv2, "invertMaps"):
        inv1, inv2 = cv2.invertMaps(map1, map2, inverseMapType=cv2.CV_32FC1)
        return inv1, inv2

    # Fallback: manual inversion with nearest-neighbor fill
    h, w = map1.shape[:2]

    inv1 = np.full((h, w), np.nan, dtype=np.float32)
    inv2 = np.full((h, w), np.nan, dtype=np.float32)

    # Source coordinates
    xs = np.tile(np.arange(w, dtype=np.float32)[None, :], (h, 1))
    ys = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))

    # Round target coordinates
    xi = np.round(map1).astype(np.int32)
    yi = np.round(map2).astype(np.int32)

    # Valid mask
    valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    inv1[yi[valid], xi[valid]] = xs[valid]
    inv2[yi[valid], xi[valid]] = ys[valid]

    # Fill holes using distance transform
    has_valid = ~np.isnan(inv1)
    if not np.any(has_valid):
        return np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32)

    invalid_mask = (~has_valid).astype(np.uint8)
    _, labels = cv2.distanceTransformWithLabels(
        invalid_mask, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )

    # Map labels to valid pixel coordinates
    label_to_yx = np.zeros((int(labels.max()) + 1, 2), dtype=np.int32)
    ys_valid, xs_valid = np.where(has_valid)
    for y, x in zip(ys_valid.tolist(), xs_valid.tolist()):
        lab = int(labels[y, x])
        label_to_yx[lab, 0] = y
        label_to_yx[lab, 1] = x

    # Fill invalid pixels
    invalid_pixels = ~has_valid
    fill_y = label_to_yx[labels[invalid_pixels], 0]
    fill_x = label_to_yx[labels[invalid_pixels], 1]

    inv1[invalid_pixels] = inv1[fill_y, fill_x]
    inv2[invalid_pixels] = inv2[fill_y, fill_x]

    inv1 = np.nan_to_num(inv1, nan=0.0).astype(np.float32)
    inv2 = np.nan_to_num(inv2, nan=0.0).astype(np.float32)

    return inv1, inv2


# =============================================================================
# Image Remapping
# =============================================================================

def remap_image(
    image: np.ndarray,
    map1: np.ndarray,
    map2: np.ndarray,
    border_mode: int,
    interpolation: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remap image and compute invalid mask.

    @param image: (H, W, C) float32 [0..1]
    @param map1: (H, W) float32 X coordinates
    @param map2: (H, W) float32 Y coordinates
    @param border_mode: cv2 border mode
    @param interpolation: cv2 interpolation mode
    @return: (remapped_image, invalid_mask) where mask is 1 for invalid pixels
    """
    require_cv2()

    h, w = image.shape[:2]

    # Remap image
    out = cv2.remap(
        image, map1, map2,
        interpolation=interpolation,
        borderMode=border_mode,
        borderValue=0
    )

    # Compute validity mask by remapping ones
    ones = np.ones((h, w), dtype=np.float32)
    valid = cv2.remap(
        ones, map1, map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    out = np.clip(out, 0.0, 1.0)
    invalid = 1.0 - np.clip(valid, 0.0, 1.0)

    return out.astype(np.float32), invalid.astype(np.float32)


def compute_valid_bbox(
    valid_mask: np.ndarray,
    threshold: float = 0.999
) -> Optional[Tuple[int, int, int, int]]:
    """
    Compute bounding box of valid region.

    @param valid_mask: (H, W) float mask where 1 = valid
    @param threshold: threshold to consider pixel valid
    @return: (x, y, width, height) or None if no valid pixels
    """
    m = valid_mask >= threshold
    if not np.any(m):
        return None

    ys, xs = np.where(m)
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1

    return (x0, y0, x1 - x0, y1 - y0)