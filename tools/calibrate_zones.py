"""
tools/calibrate_zones.py

Small one-frame polygon calibration helper.

Usage:
    python tools/calibrate_zones.py path_or_rtsp_url

Left-click points around the road/lane area you want to trace.
Press 'n' to finish that polygon and start another one.
Press 's' to print paste-ready normalized coordinates.
Press 'q' to quit without saving.

This is deliberately standalone and simple. It does not change
thresholds.py for you; it only prints values you can inspect and paste.
"""

import sys

import cv2


WINDOW_NAME = "Zone Calibration"
LINE_COLOR = (0, 255, 0)
DOT_RADIUS = 4

frame = None
current_polygon = []
finished_polygons = []


def normalize_polygon(polygon, width, height):
    """
    Convert pixel points like (320, 240) into normalized points like
    (0.5, 0.5), which is the format thresholds.py already uses.
    """
    normalized = []

    for x, y in polygon:
        normalized.append((round(x / width, 4), round(y / height, 4)))

    return normalized


def draw_polygons():
    """
    Redraw the original frame plus all saved polygons and the current
    in-progress polygon. Redrawing from the clean frame avoids leaving
    old lines behind when the user keeps clicking.
    """
    display = frame.copy()

    for polygon in finished_polygons:
        draw_one_polygon(display, polygon, close_shape=True)

    draw_one_polygon(display, current_polygon, close_shape=False)
    cv2.imshow(WINDOW_NAME, display)


def draw_one_polygon(display, polygon, close_shape):
    """
    Draw one polygon with the same simple green-line style used by the
    dashboard overlays: small dots at points, green lines between them.
    """
    for point in polygon:
        cv2.circle(display, point, DOT_RADIUS, LINE_COLOR, -1)

    for i in range(1, len(polygon)):
        cv2.line(display, polygon[i - 1], polygon[i], LINE_COLOR, 2)

    if close_shape and len(polygon) >= 3:
        cv2.line(display, polygon[-1], polygon[0], LINE_COLOR, 2)


def on_mouse(event, x, y, flags, param):
    """Left-click adds one point to the polygon currently being traced."""
    if event == cv2.EVENT_LBUTTONDOWN:
        current_polygon.append((x, y))
        draw_polygons()


def finish_current_polygon():
    """
    Move the current polygon into the finished list if it has enough
    points to form a real area. Short accidental clicks are ignored.
    """
    if len(current_polygon) >= 3:
        finished_polygons.append(list(current_polygon))
        current_polygon.clear()
        draw_polygons()
    else:
        print("Need at least 3 points before pressing 'n'.")


def parse_flow_vector(text):
    """
    Accept simple console input like '1,0' or '-1, 0'. If parsing fails,
    fall back to (1.0, 0.0) so the printed dict is still paste-ready.
    """
    try:
        parts = text.replace(",", " ").split()
        return (float(parts[0]), float(parts[1]))
    except (IndexError, ValueError):
        print("Could not read that flow vector; using (1.0, 0.0).")
        return (1.0, 0.0)


def print_results():
    """Print both formats already used in config/thresholds.py."""
    height, width = frame.shape[:2]

    if len(current_polygon) >= 3:
        finished_polygons.append(list(current_polygon))
        current_polygon.clear()

    if not finished_polygons:
        print("No finished polygons to print.")
        return

    print("\nCONGESTION_ROI_POLYGON_NORM example:")
    print(normalize_polygon(finished_polygons[0], width, height))

    print("\nWRONG_WAY_ZONES example:")
    zones = []
    for i, polygon in enumerate(finished_polygons, start=1):
        text = input(f"Flow vector for polygon {i} (example 1,0): ")
        zones.append({
            "polygon": normalize_polygon(polygon, width, height),
            "flow_vector": parse_flow_vector(text),
        })
    print(zones)


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/calibrate_zones.py path_or_rtsp_url")
        return

    capture = cv2.VideoCapture(sys.argv[1])
    ok, loaded_frame = capture.read()
    capture.release()

    if not ok:
        print("Could not read one frame from:", sys.argv[1])
        return

    global frame
    frame = loaded_frame

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    draw_polygons()

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == ord("n"):
            finish_current_polygon()
        elif key == ord("s"):
            print_results()
            break
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
