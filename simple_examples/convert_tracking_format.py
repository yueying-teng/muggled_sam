import json
import argparse
import os

import cv2


def convert_tracking_json(input_path, output_path, fps=None):

    with open(input_path) as f:
        data = json.load(f)

    width = data["frame_size"]["width"]
    height = data["frame_size"]["height"]

    if fps is None:
        video_path = data.get("video_path", "")
        if not os.path.isabs(video_path):
            video_path = os.path.join(os.path.dirname(input_path), video_path)
        vcap = cv2.VideoCapture(video_path)
        fps = vcap.get(cv2.CAP_PROP_FPS)
        vcap.release()
        if fps <= 0:
            raise RuntimeError(f"Could not read FPS from video: {video_path}. Please specify --fps manually.")
        print(f"Read FPS from video: {fps}")

    result = []
    for frame in data["frames"]:
        frame_idx = frame["frame_index"]
        total_seconds = frame_idx / fps
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60
        timestamp = f"{minutes:02d}:{seconds:06.3f}"

        entry = {"timestamp": timestamp}
        for obj in frame["objects"]:
            oid = str(obj["object_id"])
            bbox = obj["bbox"]
            location = [
                round(bbox["x1"] / width, 2),
                round(bbox["y1"] / height, 2),
                round(bbox["x2"] / width, 2),
                round(bbox["y2"] / height, 2),
            ]
            entry[oid] = {"location": location}

        result.append(entry)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Converted {len(result)} frames -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input tracking JSON file")
    parser.add_argument("--output", default=None, help="Output JSON file (default: <input_dir>/tracking_results_converted.json)")
    parser.add_argument("--fps", type=float, default=None, help="Override FPS (default: read from video)")
    args = parser.parse_args()
    output_path = args.output
    if output_path is None:
        input_dir = os.path.dirname(args.input) or "."
        output_path = os.path.join(input_dir, "tracking_results_converted.json")
    convert_tracking_json(args.input, output_path, args.fps)

"""
uv run muggled_sam/simple_examples/convert_tracking_format.py output/250830_cau_r6/tracking/tracking_results.json
"""