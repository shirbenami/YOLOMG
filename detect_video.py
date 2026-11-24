"""Run YOLOMG end to end on an .mp4: build the motion maps, detect, write a video.

The repo ships the two halves separately — test_code/generate_mask5.py writes
motion maps to a hardcoded path, and dualdetector.py reads one pre-made pair —
so there is no path from a video file to boxes. This joins them in memory.

The motion map for frame N-2 is built from frames N-4, N-2 and N, exactly as in
test_code/FD5_mask.py: blur, grayscale, align both neighbours onto the centre
frame with motion_compensate, take absolute differences, average them. That is
why the output is 4 frames shorter than the input and lags it by 2 frames.

    python3 detect_video.py --source clip.mp4
    python3 detect_video.py --source clip.mp4 --out boxes.mp4 --side-by-side
    python3 detect_video.py --source clip.mp4 --device 0 --max-frames 300
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'test_code'))

from MOD_Functions import motion_compensate  # noqa: E402

from demo_cpu import DualDetector  # noqa: E402
from dualdetector import draw_predictions  # noqa: E402

# motion_compensate falls back to this near-identity matrix when fewer than 15
# grid points survive KLT tracking, i.e. when it failed to register the frames.
IDENTITY_FALLBACK = np.array([[0.999, 0, 0], [0, 0.999, 0], [0, 0, 1]])


def to_gray(frame):
    return cv2.cvtColor(cv2.GaussianBlur(frame, (11, 11), 0), cv2.COLOR_BGR2GRAY)


def motion_map(prev_gray, centre_gray, next_gray):
    """The FD5_mask computation, kept in memory instead of written to disk.

    Returns the map as a 3-channel uint8 image plus the number of registration
    failures (0, 1 or 2) seen while building it.
    """
    warped_prev, *_, h_prev = motion_compensate(prev_gray, centre_gray)
    warped_next, *_, h_next = motion_compensate(next_gray, centre_gray)

    diff = (cv2.absdiff(centre_gray, warped_prev).astype(np.float32)
            + cv2.absdiff(centre_gray, warped_next).astype(np.float32)) / 2
    diff = np.clip(diff, 0, 255).astype(np.uint8)

    # The training masks were JPEGs on disk, so round-trip through JPEG to feed
    # the model the same compression artefacts it was trained on.
    diff = cv2.imdecode(cv2.imencode('.jpg', diff)[1], cv2.IMREAD_GRAYSCALE)

    failures = sum(np.allclose(h, IDENTITY_FALLBACK) for h in (h_prev, h_next))
    return cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR), failures


def stack_side_by_side(frame, mask):
    """Half-width frame and motion map next to each other, for eyeballing registration."""
    h, w = frame.shape[:2]
    scale = 0.5
    small = (int(w * scale), int(h * scale))
    pair = np.hstack([cv2.resize(frame, small), cv2.resize(mask, small)])
    cv2.putText(pair, 'detections', (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(pair, 'motion map', (small[0] + 12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return pair


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', required=True, help='input video')
    p.add_argument('--out', default=None, help='output video (default: <source>_yolomg.mp4)')
    p.add_argument('--weights', default=ROOT / 'runs/train/ARD100_mask32-1280_uavs/weights/best.pt')
    p.add_argument('--device', default='cpu', help="'cpu', or a CUDA index like '0'")
    p.add_argument('--imgsz', type=int, default=1280)
    p.add_argument('--conf-thres', type=float, default=0.25)
    p.add_argument('--iou-thres', type=float, default=0.4)
    p.add_argument('--max-frames', type=int, default=0, help='stop early; 0 means the whole video')
    p.add_argument('--side-by-side', action='store_true',
                   help='write detections and motion map next to each other')
    p.add_argument('--save-masks', default=None, help='also dump the motion maps into this directory')
    return p.parse_args()


def main():
    args = parse_args()

    source = Path(args.source)
    if not source.exists():
        raise SystemExit(f'no such file: {source}')

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise SystemExit(f'could not open {source} — is it a video OpenCV can read?')

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'{source.name}: {w}x{h} @ {fps:.2f} fps, {n_total} frames')

    out_path = Path(args.out) if args.out else source.with_name(source.stem + '_yolomg.mp4')
    out_size = (w, h) if not args.side_by_side else (2 * (w // 2), h // 2)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, out_size)

    masks_dir = Path(args.save_masks) if args.save_masks else None
    if masks_dir:
        masks_dir.mkdir(parents=True, exist_ok=True)

    detector = DualDetector(args.weights, device=args.device, imgsz=args.imgsz)
    print(f'model on {detector.device.type}, classes {detector.names}\n')

    # frames[0] is N-4 and frames[2] is the centre frame once five are buffered
    frames = deque(maxlen=5)
    idx = 0
    written = 0
    n_with_det = 0
    n_det = 0
    reg_failures = 0
    t_mask = 0.0
    t_infer = 0.0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        frames.append(frame)
        if len(frames) < 5:
            continue

        centre = frames[2]

        t = time.perf_counter()
        mask, failures = motion_map(to_gray(frames[0]), to_gray(centre), to_gray(frames[4]))
        t_mask += time.perf_counter() - t
        reg_failures += failures

        t = time.perf_counter()
        labels, scores, boxes = detector.run(centre, mask,
                                             conf_thres=args.conf_thres,
                                             iou_thres=args.iou_thres)
        t_infer += time.perf_counter() - t

        annotated = centre.copy()
        for label, score, box in zip(labels, scores, boxes):
            annotated = draw_predictions(annotated, label, score, box)

        if labels:
            n_with_det += 1
            n_det += len(labels)

        if masks_dir:
            cv2.imwrite(str(masks_dir / f'{idx - 2:06d}.jpg'), mask)

        writer.write(stack_side_by_side(annotated, mask) if args.side_by_side else annotated)
        written += 1

        if written % 10 == 0 or written == 1:
            rate = written / (time.time() - t0)
            print(f'\r  frame {idx - 2}  written {written}  '
                  f'detections {n_det}  {rate:.2f} fps', end='', flush=True)

        if args.max_frames and written >= args.max_frames:
            break

    cap.release()
    writer.release()

    if not written:
        raise SystemExit('\nvideo too short — at least 5 frames are needed to build one motion map')

    print(f'\n\nwrote {out_path}  ({written} frames)')
    print(f'  frames with a detection : {n_with_det}/{written} ({100*n_with_det/written:.1f}%)')
    print(f'  total detections        : {n_det}')
    print(f'  motion map              : {1000*t_mask/written:.1f} ms/frame')
    print(f'  inference               : {1000*t_infer/written:.1f} ms/frame')
    print(f'  overall                 : {written/(time.time()-t0):.2f} fps')

    if reg_failures:
        pct = 100 * reg_failures / (2 * written)
        print(f'\n  WARNING: registration fell back to identity on {reg_failures} of '
              f'{2*written} alignments ({pct:.1f}%).')
        print('  Those frames got no motion compensation, so their motion map is camera')
        print('  motion rather than object motion. Re-run with --side-by-side to look at it.')


if __name__ == '__main__':
    main()
