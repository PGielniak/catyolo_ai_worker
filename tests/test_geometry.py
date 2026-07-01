import pytest
from detector.geometry import (
    box_polygon_intersection_area,
    point_in_polygon,
    polygon_area,
    polygon_bounding_box,
    rect_to_polygon,
    zone_to_polygon,
)


def test_polygon_area_square():
    poly = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert polygon_area(poly) == 100.0


def test_point_in_polygon():
    poly = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert point_in_polygon((5, 5), poly) is True
    assert point_in_polygon((15, 5), poly) is False
    assert point_in_polygon((0, 0), poly) is True


def test_box_polygon_intersection_full_overlap():
    box = (2, 2, 8, 8)
    poly = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert box_polygon_intersection_area(box, poly) == 36.0


def test_box_polygon_intersection_partial():
    box = (5, 5, 15, 15)
    poly = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert box_polygon_intersection_area(box, poly) == 25.0


def test_box_polygon_intersection_no_overlap():
    box = (20, 20, 30, 30)
    poly = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert box_polygon_intersection_area(box, poly) == 0.0


def test_polygon_bounding_box():
    poly = [(1, 2), (5, 1), (4, 6), (0, 5)]
    assert polygon_bounding_box(poly) == (0, 1, 5, 6)


def test_rect_to_polygon():
    assert rect_to_polygon(1, 2, 3, 4) == [(1, 2), (4, 2), (4, 6), (1, 6)]


def test_zone_to_polygon_from_points():
    rz = {"points": [[0, 0], [10, 0], [10, 10], [0, 10]]}
    assert zone_to_polygon(rz) == [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]


def test_zone_to_polygon_legacy_rect():
    rz = {"x": 1, "y": 2, "width": 3, "height": 4}
    assert zone_to_polygon(rz) == [(1.0, 2.0), (4.0, 2.0), (4.0, 6.0), (1.0, 6.0)]


def test_triangle_box_intersection():
    poly = [(0, 0), (10, 0), (5, 10)]
    box = (0, 0, 10, 5)
    # The box cuts the triangle horizontally at y=5; the clipped polygon is a trapezoid
    area = box_polygon_intersection_area(box, poly)
    assert area == 37.5
