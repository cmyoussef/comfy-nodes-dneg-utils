# src\comfyui_remote\custom_nodes\wan_helper_nodes\alembic_locator.py
"""
Alembic file utilities for reading locator and joint transforms.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np

# Import VFX Alembic via alembicAbc wrapper (avoids clash with SQLAlchemy's alembic)
try:
    from alembicAbc import alembic as _abc
    Abc = _abc.Abc
    AbcGeom = _abc.AbcGeom
    HAS_ALEMBIC = True
except ImportError:
    HAS_ALEMBIC = False


def require_alembic():
    """Raise an error if VFX Alembic bindings are not available."""
    if not HAS_ALEMBIC:
        raise ImportError(
            "VFX Alembic Python bindings (alembicAbc) are required. "
            "Install the alembicAbc package that provides alembic.so."
        )


@dataclass
class LocatorTransform:
    """Represents a locator or joint transform at a specific time."""

    name: str
    position: np.ndarray  # [x, y, z]
    rotation_matrix: np.ndarray  # 3x3 rotation matrix
    full_matrix: np.ndarray  # 4x4 transformation matrix
    time: float


def _get_xform_matrix(xform_schema, time_sample) -> np.ndarray:
    """
    Extract the 4x4 transformation matrix from an Alembic XformSchema.

    Args:
        xform_schema: AbcGeom.IXformSchema
        time_sample: Abc.ISampleSelector for the time

    Returns:
        4x4 numpy array representing the world transformation matrix
    """
    sample = xform_schema.getValue(time_sample)
    mat = sample.getMatrix()

    result = np.zeros((4, 4), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            result[i, j] = mat[i][j]

    return result


def _find_object_recursive(obj, target_name: str, results: List, current_path: str = ""):
    """
    Recursively search for objects matching the target name.

    Args:
        obj: Current Alembic object
        target_name: Name to search for (can be partial match)
        results: List to append found objects to
        current_path: Current hierarchy path
    """
    name = obj.getName()
    full_path = f"{current_path}/{name}" if current_path else name

    if target_name.lower() in name.lower():
        results.append((obj, full_path))

    for i in range(obj.getNumChildren()):
        child = obj.getChild(i)
        _find_object_recursive(child, target_name, results, full_path)


def _compute_world_matrix(obj, time_sample) -> np.ndarray:
    """
    Compute the world transformation matrix by traversing up the hierarchy.

    Args:
        obj: Alembic object
        time_sample: Time sample selector

    Returns:
        4x4 world transformation matrix
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


def list_locators_in_abc(abc_path: str) -> List[str]:
    """
    List all transform nodes in an Alembic file.

    Args:
        abc_path: Path to the Alembic file

    Returns:
        List of object paths found in the file
    """
    require_alembic()

    archive = Abc.IArchive(abc_path)
    top = archive.getTop()

    results = []

    def collect_all(obj, path=""):
        name = obj.getName()
        full_path = f"{path}/{name}" if path else name

        if name:
            header = obj.getHeader()
            if AbcGeom.IXform.matches(header):
                results.append(full_path)

        for i in range(obj.getNumChildren()):
            collect_all(obj.getChild(i), full_path)

    collect_all(top)
    return results


def get_locator_transform(
    abc_path: str,
    locator_name: str,
    frame: int,
    fps: float = 24.0,
) -> LocatorTransform:
    """
    Get the world transform of a locator or joint from an Alembic file.

    Args:
        abc_path: Path to the Alembic file
        locator_name: Name of the locator or joint to find (partial match supported)
        frame: Frame number to sample
        fps: Frames per second for time conversion

    Returns:
        LocatorTransform containing position and rotation

    Raises:
        ValueError: If locator not found or multiple matches
    """
    require_alembic()

    archive = Abc.IArchive(abc_path)
    top = archive.getTop()

    matches = []
    _find_object_recursive(top, locator_name, matches)

    if not matches:
        available = list_locators_in_abc(abc_path)
        raise ValueError(
            f"Locator '{locator_name}' not found in {abc_path}. "
            f"Available objects: {available[:10]}{'...' if len(available) > 10 else ''}"
        )

    exact_matches = [
        (obj, path) for obj, path in matches if obj.getName().lower() == locator_name.lower()
    ]
    if exact_matches:
        matches = exact_matches

    if len(matches) > 1:
        paths = [path for _, path in matches]
        raise ValueError(
            f"Multiple objects match '{locator_name}': {paths}. "
            "Please provide a more specific name."
        )

    obj, full_path = matches[0]

    time = frame / fps
    time_sample = Abc.ISampleSelector(time)

    world_matrix = _compute_world_matrix(obj, time_sample)
    position = world_matrix[:3, 3].copy()
    rotation_matrix = world_matrix[:3, :3].copy()

    return LocatorTransform(
        name=full_path,
        position=position,
        rotation_matrix=rotation_matrix,
        full_matrix=world_matrix,
        time=time,
    )


def get_locator_transforms_batch(
    abc_path: str,
    locator_name: str,
    frames: List[int],
    fps: float = 24.0,
) -> List[LocatorTransform]:
    """
    Get transforms for multiple frames efficiently.

    Args:
        abc_path: Path to the Alembic file
        locator_name: Name of the locator or joint
        frames: List of frame numbers
        fps: Frames per second

    Returns:
        List of LocatorTransform objects, one per frame
    """
    require_alembic()

    archive = Abc.IArchive(abc_path)
    top = archive.getTop()

    matches = []
    _find_object_recursive(top, locator_name, matches)

    if not matches:
        available = list_locators_in_abc(abc_path)
        raise ValueError(
            f"Locator '{locator_name}' not found. Available: {available[:10]}"
        )

    exact_matches = [
        (obj, path) for obj, path in matches if obj.getName().lower() == locator_name.lower()
    ]
    if exact_matches:
        matches = exact_matches

    if len(matches) > 1:
        paths = [path for _, path in matches]
        raise ValueError(f"Multiple matches for '{locator_name}': {paths}")

    obj, full_path = matches[0]

    results = []
    for frame in frames:
        time = frame / fps
        time_sample = Abc.ISampleSelector(time)

        world_matrix = _compute_world_matrix(obj, time_sample)
        position = world_matrix[:3, 3].copy()
        rotation_matrix = world_matrix[:3, :3].copy()

        results.append(
            LocatorTransform(
                name=full_path,
                position=position,
                rotation_matrix=rotation_matrix,
                full_matrix=world_matrix,
                time=time,
            )
        )

    return results
