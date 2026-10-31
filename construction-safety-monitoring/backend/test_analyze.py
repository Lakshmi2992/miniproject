"""
Runs the REAL analyze_video() function (the one your Flask app actually
calls) and prints its decision — not just raw angles like debug_fall.py.
Usage:  python test_analyze.py uploads/falldetected_video.mp4
"""
import sys
import json
from detect import analyze_video

result = analyze_video(sys.argv[1])

print("frames_analyzed:", result["frames_analyzed"])
print("fall_detected:", result["fall_detected"])
print("fall_count:", result["fall_count"])
print("\nfall_events:")
for e in result["fall_events"]:
    print(" ", e)