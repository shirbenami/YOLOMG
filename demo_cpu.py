"""Run the YOLOMG dual-branch detector on a single image + motion-mask pair.

A device-agnostic companion to dualdetector.py, which hardcodes select_device('0')
and therefore needs a CUDA GPU. This one takes --device, defaults to CPU, and
rescales boxes against the real image size instead of a hardcoded 1080x1920.

    python3 demo_cpu.py                       # the bundled phantom05 sample
    python3 demo_cpu.py --device 0            # same, on a CUDA GPU
    python3 demo_cpu.py --image a.jpg --mask a_mask.jpg --out pred.jpg
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from dualdetector import draw_predictions
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device

ROOT = Path(__file__).resolve().parent


class DualDetector:
    """YOLOMG inference: an RGB frame and its motion difference map in, boxes out."""

    def __init__(self, weights, device='cpu', imgsz=1280):
        self.device = select_device(device)
        # FP16 is a CUDA-only path; CPU must stay FP32.
        self.half = self.device.type != 'cpu'

        self.model = attempt_load(str(weights), map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.imgsz = check_img_size(imgsz, s=self.stride)
        if self.half:
            self.model.half()

        self.names = getattr(self.model, 'module', self.model).names

    def _preprocess(self, img):
        img = letterbox(img, self.imgsz, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        return img.unsqueeze(0) if img.ndimension() == 3 else img

    def run(self, rgb, mask, conf_thres=0.1, iou_thres=0.4, classes=None):
        img1 = self._preprocess(rgb)
        img2 = self._preprocess(mask)

        with torch.no_grad():
            pred = self.model(img1, img2, augment=False)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=False)

        det = pred[0]
        if not len(det):
            return [], [], np.empty((0, 4))

        # Map boxes back onto the original frame, not a hardcoded resolution.
        boxes = scale_coords(img1.shape[2:], det[:, :4], rgb.shape).round().cpu().numpy()
        labels = [self.names[int(cls)] for cls in det[:, -1]]
        scores = [round(float(conf), 2) for conf in det[:, -2]]
        return labels, scores, boxes


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--weights', default=ROOT / 'runs/train/ARD100_mask32-1280_uavs/weights/best.pt',
                   help='checkpoint; the 1280 ARD100 model committed to this repo by default')
    p.add_argument('--image', default=ROOT / 'data/Test_images/images/phantom05_0606.jpg',
                   help='RGB frame')
    p.add_argument('--mask', default=ROOT / 'data/Test_images/mask/phantom05_0606.jpg',
                   help='matching motion difference map')
    p.add_argument('--device', default='cpu', help="'cpu', or a CUDA index like '0'")
    p.add_argument('--imgsz', type=int, default=1280)
    p.add_argument('--conf-thres', type=float, default=0.1)
    p.add_argument('--iou-thres', type=float, default=0.4)
    p.add_argument('--out', default=None, help='write an annotated copy of --image here')
    return p.parse_args()


def main():
    args = parse_args()

    rgb = cv2.imread(str(args.image))
    mask = cv2.imread(str(args.mask))
    if rgb is None:
        raise SystemExit(f'could not read image: {args.image}')
    if mask is None:
        raise SystemExit(f'could not read mask: {args.mask}')
    if rgb.shape != mask.shape:
        raise SystemExit(f'image {rgb.shape} and mask {mask.shape} must have the same shape')

    print(f'weights: {args.weights}')
    print(f'image:   {args.image}  {rgb.shape[1]}x{rgb.shape[0]}')
    detector = DualDetector(args.weights, device=args.device, imgsz=args.imgsz)
    print(f'classes: {detector.names}\n')

    t0 = time.time()
    labels, scores, boxes = detector.run(rgb, mask,
                                         conf_thres=args.conf_thres,
                                         iou_thres=args.iou_thres)
    elapsed = time.time() - t0

    print(f'inference: {elapsed:.2f}s on {detector.device.type}')
    print(f'detections: {len(labels)}')
    for label, score, box in zip(labels, scores, boxes):
        u1, v1, u2, v2 = (int(v) for v in box)
        print(f'  {label:10s} conf={score:.2f}  xyxy=({u1}, {v1}, {u2}, {v2})  '
              f'{u2 - u1}x{v2 - v1}px')

    if args.out:
        annotated = rgb.copy()
        for label, score, box in zip(labels, scores, boxes):
            annotated = draw_predictions(annotated, label, score, box)
        cv2.imwrite(str(args.out), annotated)
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
