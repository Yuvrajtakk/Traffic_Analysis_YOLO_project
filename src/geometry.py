"""
src/geometry.py

Small shared geometry helpers for any analytics module that needs
point-in-polygon testing against a normalized (0-1) ROI/zone polygon.

Pulled out into one place so congestion.py's ROI check and
wrong_way.py's per-zone check share ONE implementation instead of two
separate copies that could quietly drift apart over time (e.g. one
gets a bugfix, the other doesn't).
"""

import cv2
import numpy as np


def denormalize_polygon(polygon_norm, frame_width, frame_height):
    """
    polygon_norm: list of (nx, ny) tuples in 0-1 normalized coordinates.
    Returns an np.int32 array in real pixel coordinates — the exact
    format cv2.pointPolygonTest (and cv2 drawing functions) expect.

    Convert ONCE per polygon, right after the video's real
    frame_width/frame_height become known — they never change for a
    given video, so there's no reason to redo this conversion every
    single frame.
    """
    return np.array(
        [(nx * frame_width, ny * frame_height) for (nx, ny) in polygon_norm],
        dtype=np.int32,
    )


def point_in_polygon(point, polygon_px):
    """
    point: (x, y) in real pixel coordinates.
    polygon_px: np.int32 array, already in pixel coordinates (the
    output of denormalize_polygon() above).

    cv2.pointPolygonTest returns: positive = inside, negative =
    outside, zero = exactly on the edge. We only care about "clearly
    inside," so anything <= 0 counts as outside.
    """
    return cv2.pointPolygonTest(
        polygon_px, (float(point[0]), float(point[1])), False
    ) > 0
