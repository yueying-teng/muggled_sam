#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This is a hack to make this script work from outside the root project folder (without requiring install)
try:
    import muggled_sam  # NOQA
except ModuleNotFoundError:
    import os
    import sys

    parent_folder = os.path.dirname(os.path.dirname(__file__))
    if "muggled_sam" in os.listdir(parent_folder):
        sys.path.insert(0, parent_folder)
    else:
        raise ImportError("Can't find path to muggled_sam folder!")
import json
from time import perf_counter
from collections import defaultdict
import cv2
import numpy as np
import torch
from muggled_sam.make_sam import make_sam_from_state_dict
from muggled_sam.demo_helpers.video_data_storage import SAMVideoObjectResults
from muggled_sam.demo_helpers.bounding_boxes import get_2box_iou, get_one_mask_bounding_box

# Define pathing & device usage
initial_frame_index = 0
video_path = "/home/ubuntu/muggled_sam/250830_cau_r6.mp4"
detection_model_path = "/home/ubuntu/muggled_sam/model_weights/sam3.pt"
tracking_model_path = None  # Can use a SAMv2 model! Leave as None to re-use the detection model for tracking
device, dtype = "cpu", torch.float32
if torch.cuda.is_available():
    device, dtype = "cuda", torch.bfloat16

# All coordinates are normalized between 0 and 1. Top left of image is (0,0), bottom-right is (1,1)
pos_box_xy1xy2_norm_list = []  # Format is: [[(x1, y1), (x2, y2)]]
neg_box_xy1xy2_norm_list = []
pos_point_xy_norm_list = []  # Format is [(x1, y1)]
neg_point_xy_norm_list = []
text_prompt = "horse"

# Controls for detection/tracking
detect_every_n_frames = 24  # Set to None to only run once on startup
detection_score_threshold = 0.8
existing_box_iou_threshold = 0.15
tracking_score_threshold = 0.0
remove_after_n_missed_frames = 15

output_dir = "output/250830_cau_r6/tracking"
os.makedirs(output_dir, exist_ok=True)
output_json_path = os.path.join(output_dir, "tracking_results.json")

track_imgenc_config_dict = {"max_side_length": None, "use_square_sizing": True}
detection_imgenc_config_dict = {"max_side_length": None, "use_square_sizing": True}
detection_prompts_dict = {
    "text": text_prompt,
    "box_xy1xy2_norm_list": pos_box_xy1xy2_norm_list,
    "point_xy_norm_list": pos_point_xy_norm_list,
    "negative_boxes_list": neg_box_xy1xy2_norm_list,
    "negative_points_list": neg_point_xy_norm_list,
}

all_frame_results = []

vcap = cv2.VideoCapture(video_path)
vcap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
vcap.set(cv2.CAP_PROP_POS_FRAMES, initial_frame_index)
ok_frame, first_frame = vcap.read()
assert ok_frame, f"Could not read frames from video: {video_path}"
frame_h, frame_w = first_frame.shape[:2]

# Set up model
print("Loading model...")
model_config_dict, track_model = make_sam_from_state_dict(detection_model_path)
assert track_model.name == "samv3", "Error! Only SAMv3 models support object detection..."
track_model.to(device=device, dtype=dtype)
detmodel = track_model.make_detector_model()
print("  Done!")

# Allow loading of alternate tracking model
if tracking_model_path is not None:
    print("Loading separate tracking model...")
    _, track_model = make_sam_from_state_dict(tracking_model_path)
    assert track_model.name in ("samv2", "samv3"), "Only SAMv2/v3 are supported for video tracking"
    track_model.to(device=device, dtype=dtype)
    print("  Done!")

# Set up storage for tracking memory and keeping track of lost objects
memory_per_obj_dict = defaultdict(SAMVideoObjectResults.create)
missed_frames_per_obj_dict = defaultdict(int)
last_known_box_per_obj = {}

detect_every_n_frames = 2**31 if detect_every_n_frames is None else detect_every_n_frames
try:
    is_webcam = isinstance(video_path, int)
    total_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else 100_000
    for idx_frame in range(initial_frame_index, total_frames):

        ok_frame, frame = vcap.read()
        if not ok_frame:
            break

        # Start model inference timing
        t1 = perf_counter()

        # Encode image data for tracking (this is the heaviest part of video inference)
        encoded_imgs_list, _, _ = track_model.encode_image(frame, **track_imgenc_config_dict)

        objs_to_remove_list = []
        masks_on_frame_list = []
        mask_obj_id_list = []
        frame_objects = []
        tracked_obj_ids = list(memory_per_obj_dict.keys())
        for idx_obj in tracked_obj_ids:
            obj_memory = memory_per_obj_dict[idx_obj]
            obj_score, best_mask_idx, mask_preds, mem_enc, obj_ptr = track_model.step_video_masking(
                encoded_imgs_list, **obj_memory.to_dict()
            )

            obj_score = obj_score.item()
            if obj_score < tracking_score_threshold:
                missed_frames_per_obj_dict[idx_obj] += 1
                if missed_frames_per_obj_dict[idx_obj] > remove_after_n_missed_frames:
                    objs_to_remove_list.append(idx_obj)
                continue
            missed_frames_per_obj_dict[idx_obj] = 0

            obj_memory.store_frame_result(idx_frame, mem_enc, obj_ptr)
            obj_mask = mask_preds[0, best_mask_idx, :, :]
            masks_on_frame_list.append(obj_mask)
            mask_obj_id_list.append(idx_obj)

            is_valid, (xy1, xy2) = get_one_mask_bounding_box(obj_mask)
            if is_valid:
                mask_h, mask_w = obj_mask.squeeze().shape
                scale_x, scale_y = frame_w / mask_w, frame_h / mask_h
                box_xy1xy2_norm = torch.tensor((
                    (float(xy1[0]) / mask_w, float(xy1[1]) / mask_h),
                    (float(xy2[0]) / mask_w, float(xy2[1]) / mask_h),
                ))
                last_known_box_per_obj[idx_obj] = box_xy1xy2_norm
                frame_objects.append({
                    "object_id": int(idx_obj),
                    "score": round(obj_score, 4),
                    "bbox": {
                        "x1": round(float(xy1[0] * scale_x), 1),
                        "y1": round(float(xy1[1] * scale_y), 1),
                        "x2": round(float(xy2[0] * scale_x), 1),
                        "y2": round(float(xy2[1] * scale_y), 1),
                    },
                })

        for idx_obj in objs_to_remove_list:
            memory_per_obj_dict.pop(idx_obj)
            missed_frames_per_obj_dict.pop(idx_obj)
            last_known_box_per_obj.pop(idx_obj, None)
            print("  -> Removed object:", idx_obj)

        no_tracked_objects = len(memory_per_obj_dict) == 0
        need_detection = ((idx_frame - initial_frame_index) % detect_every_n_frames) == 0 or no_tracked_objects
        if need_detection:
            print(f"  Performing detection update! (frame {idx_frame})")
            det_encimgs, _, _ = detmodel.encode_detection_image(frame, **detection_imgenc_config_dict)
            det_exemplars = detmodel.encode_exemplars(det_encimgs, **detection_prompts_dict)
            det_masks, det_boxes, _, _ = detmodel.generate_detections(
                det_encimgs, det_exemplars, detection_filter_threshold=detection_score_threshold
            )

            # If we get new detections, compare to existing objects to see if anything new has appeared
            num_detections = det_masks.shape[1]
            print(f"    -> Detected {num_detections} objects")
            if num_detections > 0:

                known_boxes_list = []
                matched_obj_ids = set()
                for mask_tensor, obj_id in zip(masks_on_frame_list, mask_obj_id_list):
                    if obj_id in [o for o in objs_to_remove_list]:
                        continue
                    mask_uint8 = (mask_tensor[0] > 0).byte().cpu().numpy()
                    contours_list, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if len(contours_list) == 0:
                        continue
                    contour = max(contours_list, key=cv2.contourArea) if len(contours_list) > 1 else contours_list[0]
                    box_x, box_y, box_w, box_h = cv2.boundingRect(contour)
                    box_xy1xy2_px = torch.tensor(((box_x, box_y), (box_x + box_w, box_y + box_h)))
                    box_xy1xy2_norm = box_xy1xy2_px / torch.tensor((mask_uint8.shape[1], mask_uint8.shape[0]))
                    known_boxes_list.append(box_xy1xy2_norm.to(det_boxes))
                    matched_obj_ids.add(obj_id)
                for obj_id, box_norm in last_known_box_per_obj.items():
                    if obj_id not in matched_obj_ids and obj_id in memory_per_obj_dict:
                        known_boxes_list.append(box_norm.to(det_boxes))

                is_new_obj_list = []
                for idx_det in range(num_detections):
                    new_box = det_boxes[0, idx_det]
                    is_known = any(get_2box_iou(new_box, b) > existing_box_iou_threshold for b in known_boxes_list)
                    is_new_obj_list.append(not is_known)

                all_used_ids = set(memory_per_obj_dict.keys()) | set(last_known_box_per_obj.keys())
                next_new_idx = max(all_used_ids) + 1 if len(all_used_ids) > 0 else 0
                new_det_idxs_list = [det_idx for det_idx, is_new in enumerate(is_new_obj_list) if is_new]
                print(f"    -> Adding {len(new_det_idxs_list)} new objects")
                for idx_offset, det_idx in enumerate(new_det_idxs_list):
                    raw_det_mask = det_masks[0, det_idx]
                    init_mem = track_model.initialize_from_mask(encoded_imgs_list, raw_det_mask > 0)
                    new_idx = next_new_idx + idx_offset
                    memory_per_obj_dict[new_idx].store_prompt_result(idx_frame, init_mem)
                    masks_on_frame_list.append(raw_det_mask.unsqueeze(0))
                    mask_obj_id_list.append(new_idx)
                pass

        t2 = perf_counter()
        print(f"Frame {idx_frame}: {round(1000 * (t2 - t1))} ms, {len(memory_per_obj_dict)} objects")

        all_frame_results.append({"frame_index": idx_frame, "objects": frame_objects})

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    vcap.release()
    with open(output_json_path, "w") as f:
        json.dump(
            {"video_path": video_path, "frame_size": {"width": frame_w, "height": frame_h}, "frames": all_frame_results},
            f,
            indent=2,
        )
    print(f"Saved {len(all_frame_results)} frames to: {output_json_path}")



"""
CUDA_VISIBLE_DEVICES=3 uv run simple_examples/video_segmentation_from_detections.py
"""