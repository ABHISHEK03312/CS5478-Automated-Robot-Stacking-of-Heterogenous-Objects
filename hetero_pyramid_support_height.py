"""Geometry-based relative layout for *target-free* 2/3/4-layer pyramids.

No reference goal poses or Ravens demonstrations are consumed. The planner chooses
object IDs, grasps and small pose offsets, while this helper enforces a requested
triangular layer topology. This is an AABB-based stability proxy, not a physical
proof of long-term stable multi-support contact.
"""
import math
import numpy as np
import pybullet as p

CENTER = np.array([0.5, 0.18], dtype=float)
COUNTS = {3: 2, 6: 3, 10: 4}


def resolve_layers(n, requested=None):
    if requested is None:
        if n not in COUNTS:
            raise ValueError(f'Cannot infer layers from {n} objects (expected 3, 6, or 10)')
        return COUNTS[n]
    if requested not in (2, 3, 4) or n != requested * (requested + 1) // 2:
        raise ValueError(f'--layers={requested} requires {requested * (requested + 1) // 2} objects, got {n}')
    return requested


def row_and_col(index, layers):
    for row in range(layers):
        count = layers - row
        if index < count:
            return row, index
        index -= count
    raise IndexError('Too many placed objects for pyramid')


def row_indices(row, layers):
    start = sum(layers - r for r in range(row))
    return start, layers - row


def _pos(bodies, oid):
    return np.asarray(p.getBasePositionAndOrientation(bodies[oid])[0], dtype=float)


def _aabb(bodies, oid):
    return p.getAABB(bodies[oid])


def pitch_for(geometry):
    # Shared pitch supports mixed footprints while keeping lower blocks adjacent.
    lengths = sorted(float(item['dims'][0]) for item in geometry.values())
    return float(np.median(lengths)) + 0.003


def placement_target(placed, oid, layers, geometry, bodies):
    """Return (XY, object-center Z); derives height from actual supporting objects."""
    row, col = row_and_col(len(placed), layers)
    height = float(geometry[oid]['dims'][2])
    if row == 0:
        pitch = pitch_for(geometry)
        x = float(CENTER[0] + (col - (layers - 1) / 2) * pitch)
        return np.array([x, float(CENTER[1])]), height / 2 + 0.002
    start, _ = row_indices(row - 1, layers)
    left, right = placed[start + col], placed[start + col + 1]
    left_top = float(_aabb(bodies, left)[1][2])
    right_top = float(_aabb(bodies, right)[1][2])
    if abs(left_top - right_top) > 0.012:
        return None
    xy = (_pos(bodies, left)[:2] + _pos(bodies, right)[:2]) / 2
    return xy, max(left_top, right_top) + height / 2 + 0.002


def pyramid_stable(placed, layers, geometry, bodies, xy_tolerance=0.023):
    """Check each placed object and expected pair of supports using settled AABBs."""
    for idx, oid in enumerate(placed):
        row, col = row_and_col(idx, layers)
        pos, quat = p.getBasePositionAndOrientation(bodies[oid])
        roll, pitch, _ = p.getEulerFromQuaternion(quat)
        if not all(math.isfinite(v) for v in pos) or abs(roll) > .20 or abs(pitch) > .20:
            return False
        aabb = _aabb(bodies, oid)
        if row == 0:
            expected_x = CENTER[0] + (col - (layers - 1) / 2) * pitch_for(geometry)
            if abs(aabb[0][2]) > 0.02 or abs(pos[0] - expected_x) > xy_tolerance or abs(pos[1] - CENTER[1]) > xy_tolerance:
                return False
            continue
        start, _ = row_indices(row - 1, layers)
        left, right = placed[start + col], placed[start + col + 1]
        top_l = _aabb(bodies, left)[1][2]
        top_r = _aabb(bodies, right)[1][2]
        if abs(top_l - top_r) > .012 or abs(aabb[0][2] - max(top_l, top_r)) > .018:
            return False
        mid = (_pos(bodies, left)[:2] + _pos(bodies, right)[:2]) / 2
        if np.linalg.norm(np.asarray(pos[:2]) - mid) > xy_tolerance:
            return False
        # An upper object must intersect both lower support footprints in XY.
        for lower in (left, right):
            lo = _aabb(bodies, lower)
            overlap_x = min(aabb[1][0], lo[1][0]) - max(aabb[0][0], lo[0][0])
            overlap_y = min(aabb[1][1], lo[1][1]) - max(aabb[0][1], lo[0][1])
            if overlap_x < .001 or overlap_y < .001:
                return False
    return True


def height_completion_possible(placed, remaining, layers, geometry, tolerance=0.004):
    """Nominal-height lookahead for a complete triangular pyramid.

    Uses each object's actual metadata height, not fixed goal heights. All
    objects in the partial ordering are fixed. Rejects completions when two
    blocks supporting the same upper block have incompatible top levels.
    A necessary kinematic test only; PyBullet validates actual placements.
    """
    from functools import lru_cache

    placed = tuple(map(int, placed))
    remaining = tuple(map(int, remaining))
    heights = {int(i): float(g['dims'][2]) for i, g in geometry.items()}
    total = layers * (layers + 1) // 2
    if len(placed) + len(remaining) != total:
        return False

    @lru_cache(maxsize=None)
    def search(order, unused):
        idx = len(order)
        if not unused:
            return True
        row, col = row_and_col(idx, layers)
        for oid in unused:
            if row > 0:
                # Nominal top heights for all previously selected objects.
                tops = []
                for j, prior in enumerate(order):
                    r, c = row_and_col(j, layers)
                    if r == 0:
                        tops.append(heights[prior])
                    else:
                        start, _ = row_indices(r - 1, layers)
                        left, right = tops[start + c], tops[start + c + 1]
                        if abs(left - right) > tolerance:
                            break
                        tops.append(max(left, right) + heights[prior])
                if len(tops) != idx:
                    return False
                start, _ = row_indices(row - 1, layers)
                if abs(tops[start + col] - tops[start + col + 1]) > tolerance:
                    continue
            rest = tuple(x for x in unused if x != oid)
            if search(order + (oid,), rest):
                return True
        return False

    return search(placed, remaining)
