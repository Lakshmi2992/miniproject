"""
Standalone diagnostic — shows exactly what the pose model detects,
frame by frame, without going through Flask at all.
Usage:  python debug_fall.py uploads/your_video.mp4
"""
import sys
import cv2
from detect import _run_pose, _torso_angle_from_vertical

def debug_video(path, frame_skip=2):
    cap = cv2.VideoCapture(path)
    frame_idx = 0
    any_people_found = False
    any_angle_found = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_skip == 0:
            people = _run_pose(frame)
            if not people:
                print(f"frame {frame_idx}: NO people detected")
            else:
                any_people_found = True
                for i, p in enumerate(people):
                    angle = _torso_angle_from_vertical(p["keypoints"])
                    if angle is not None:
                        any_angle_found = True
                    print(f"frame {frame_idx}: person {i} -> angle={angle}, box={p['box']}")
        frame_idx += 1

    cap.release()
    print("\n--- SUMMARY ---")
    print("Any people ever detected:", any_people_found)
    print("Any torso angle ever calculated:", any_angle_found)

if __name__ == "__main__":
    debug_video(sys.argv[1])