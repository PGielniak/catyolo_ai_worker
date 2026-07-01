"""Pure-geometry helpers for polygon red zones.

All coordinates are in pixel space. Polygons are lists of (x, y) vertices in
clockwise or counter-clockwise order. Boxes are (x1, y1, x2, y2).
"""

from typing import List, Sequence, Tuple

Point = Tuple[float, float]
Polygon = Sequence[Point]
Box = Tuple[float, float, float, float]


def _to_point(value) -> Point:
    return (float(value[0]), float(value[1]))


def polygon_area(poly: Polygon) -> float:
    if len(poly) < 3:
        return 0.0
    area = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def point_in_polygon(p: Point, poly: Polygon) -> bool:
    if len(poly) < 3:
        return False
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            inside = not inside
    return inside


def polygon_bounding_box(poly: Polygon) -> Box:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (min(xs), min(ys), max(xs), max(ys))


def rect_to_polygon(x: float, y: float, w: float, h: float) -> List[Point]:
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def zone_to_polygon(rz: dict) -> List[Point]:
    """Return a polygon for a red-zone dict, converting legacy rectangles."""
    raw_points = rz.get("points")
    if raw_points and len(raw_points) >= 3:
        return [_to_point(p) for p in raw_points]
    x = rz.get("x", 0) or 0
    y = rz.get("y", 0) or 0
    w = rz.get("width", 0) or 0
    h = rz.get("height", 0) or 0
    return rect_to_polygon(float(x), float(y), float(w), float(h))


def _inside(p: Point, edge: Tuple[Point, Point], side: str) -> bool:
    (x1, y1), (x2, y2) = edge
    x, y = p
    if side == "left":
        return x >= x1
    if side == "right":
        return x <= x2
    if side == "bottom":
        return y >= y1
    if side == "top":
        return y <= y2
    raise ValueError(f"Unknown side: {side}")


def _intersect(a: Point, b: Point, edge: Tuple[Point, Point], side: str) -> Point:
    (x1, y1), (x2, y2) = edge
    ax, ay = a
    bx, by = b
    if side == "left":
        x = x1
        t = (x - ax) / (bx - ax) if bx != ax else 0.0
        return (x, ay + t * (by - ay))
    if side == "right":
        x = x2
        t = (x - ax) / (bx - ax) if bx != ax else 0.0
        return (x, ay + t * (by - ay))
    if side == "bottom":
        y = y1
        t = (y - ay) / (by - ay) if by != ay else 0.0
        return (ax + t * (bx - ax), y)
    if side == "top":
        y = y2
        t = (y - ay) / (by - ay) if by != ay else 0.0
        return (ax + t * (bx - ax), y)
    raise ValueError(f"Unknown side: {side}")


def _sutherland_hodgman(
    subject: List[Point], edge: Tuple[Point, Point], side: str
) -> List[Point]:
    output: List[Point] = []
    n = len(subject)
    if n == 0:
        return output
    for i in range(n):
        current = subject[i]
        previous = subject[i - 1]
        cur_in = _inside(current, edge, side)
        prev_in = _inside(previous, edge, side)
        if cur_in:
            if not prev_in:
                output.append(_intersect(previous, current, edge, side))
            output.append(current)
        elif prev_in:
            output.append(_intersect(previous, current, edge, side))
    return output


def box_polygon_intersection_area(box: Box, poly: Polygon) -> float:
    """Area of overlap between an axis-aligned box and a polygon."""
    if len(poly) < 3:
        return 0.0
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return 0.0
    edges = [
        ("left", ((x1, y1), (x1, y2))),
        ("right", ((x2, y1), (x2, y2))),
        ("bottom", ((x1, y1), (x2, y1))),
        ("top", ((x1, y2), (x2, y2))),
    ]
    output: List[Point] = list(poly)
    for side, edge in edges:
        output = _sutherland_hodgman(output, edge, side)
        if not output:
            return 0.0
    return polygon_area(output)
