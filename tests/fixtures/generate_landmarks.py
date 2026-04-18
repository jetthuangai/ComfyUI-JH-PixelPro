from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp

LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
RIGHT_EYE_OUTER = 263
RIGHT_EYE_INNER = 362
NOSE_TIP = 1
MOUTH_LEFT = 61
MOUTH_RIGHT = 291


def _point(face_landmarks, index: int, width: int, height: int) -> list[float]:
    landmark = face_landmarks.landmark[index]
    return [landmark.x * (width - 1), landmark.y * (height - 1)]


def extract_landmarks(image_path: Path) -> list[list[float]]:
    if not hasattr(mp, "solutions"):
        raise RuntimeError(
            "Installed mediapipe wheel does not expose mp.solutions in this environment. "
            "Use sample_portrait_landmarks.json fallback or switch to a compatible MediaPipe build."
        )

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]
    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
    )
    result = mesh.process(image_rgb)
    mesh.close()
    if not result.multi_face_landmarks:
        raise RuntimeError("No face landmarks detected.")

    face = result.multi_face_landmarks[0]
    left_outer = _point(face, LEFT_EYE_OUTER, width, height)
    left_inner = _point(face, LEFT_EYE_INNER, width, height)
    right_outer = _point(face, RIGHT_EYE_OUTER, width, height)
    right_inner = _point(face, RIGHT_EYE_INNER, width, height)

    return [
        [(left_outer[0] + left_inner[0]) / 2.0, (left_outer[1] + left_inner[1]) / 2.0],
        [(right_outer[0] + right_inner[0]) / 2.0, (right_outer[1] + right_inner[1]) / 2.0],
        _point(face, NOSE_TIP, width, height),
        _point(face, MOUTH_LEFT, width, height),
        _point(face, MOUTH_RIGHT, width, height),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate 5-point landmarks with MediaPipe.")
    parser.add_argument("image", type=Path, help="Input portrait image path.")
    parser.add_argument("output", type=Path, help="Output JSON path.")
    args = parser.parse_args()

    args.output.write_text(json.dumps(extract_landmarks(args.image), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
