import json
import argparse
import cv2
import numpy as np

COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 255), (255, 128, 0), (0, 128, 255), (128, 255, 0),
    (255, 0, 128), (0, 255, 128), (200, 100, 50), (50, 200, 100), (100, 50, 200),
    (64, 224, 208), (220, 20, 60), (75, 0, 130), (245, 166, 35), (34, 139, 34),
]


def get_color(obj_id):
    return COLORS[obj_id % len(COLORS)]


parser = argparse.ArgumentParser()
parser.add_argument("json_path", help="Path to tracking_results.json")
parser.add_argument("-o", "--output", default=None, help="Output video path (if not set, displays live)")
args = parser.parse_args()

with open(args.json_path, "r") as f:
    data = json.load(f)

video_path = data["video_path"]
frames_data = {f["frame_index"]: f["objects"] for f in data["frames"]}

vcap = cv2.VideoCapture(video_path)
vcap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
ok, first_frame = vcap.read()
assert ok, f"Could not read video: {video_path}"
vcap.set(cv2.CAP_PROP_POS_FRAMES, 0)

h, w = first_frame.shape[:2]
fps = vcap.get(cv2.CAP_PROP_FPS) or 24

writer = None
if args.output:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

frame_idx = 0
while True:
    ok, frame = vcap.read()
    if not ok:
        break

    objects = frames_data.get(frame_idx, [])
    for obj in objects:
        oid = obj["object_id"]
        b = obj["bbox"]
        x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
        color = get_color(oid)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"ID:{oid}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = max(y1 - 6, th + 2)
        cv2.rectangle(frame, (x1, label_y - th - 2), (x1 + tw, label_y + baseline), color, -1)
        cv2.putText(frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    if writer:
        writer.write(frame)
    else:
        cv2.imshow("Tracking Visualization - q to quit", frame)
        if (cv2.waitKey(1) & 0xFF) in {27, ord("q")}:
            break

    frame_idx += 1

vcap.release()
if writer:
    writer.release()
    print(f"Saved to: {args.output}")
else:
    cv2.destroyAllWindows()
    
    
"""
uv run simple_examples/visualize_tracking_json.py tracking_results_per_8_frames.json -o output_per_8_frames.mp4

uv run simple_examples/visualize_tracking_json.py tracking_results_per_12_frames.json -o output_per_12_frames.mp4

uv run simple_examples/visualize_tracking_json.py tracking_results_per_24_frames.json -o output_per_24_frames.mp4
"""