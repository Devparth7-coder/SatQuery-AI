"""
Build a ground-truthed bi-temporal demo pair from a real Sentinel-2 scene.

T2 applies known edits so change-detection output can be verified against truth:
  * three forest-clearing patches  (vegetation -> bare soil)
  * one urban expansion block      (vegetation -> built-up grey)
  * reservoir/inundation patch     (land -> water)
  * a +6/-4 px translation so co-registration has real work to do
"""
import json
import cv2
import numpy as np
from PIL import Image

SRC = "samples/river_wetland.png"
rgb = np.array(Image.open(SRC).convert("RGB"))
H, W = rgb.shape[:2]
rng = np.random.RandomState(7)

t1 = rgb.copy()
t2 = rgb.copy()
truth = []


def texture(shape, base, spread=14):
    n = rng.normal(0, spread, (*shape, 3))
    return np.clip(np.array(base, float) + n, 0, 255).astype(np.uint8)


def paste(mask, base_col, label, spread=14):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return
    patch = texture((H, W), base_col, spread)
    t2[mask] = patch[mask]
    truth.append({"kind": label, "pixels": int(mask.sum()),
                  "fraction": round(float(mask.mean()), 5)})


def ellipse_mask(cx, cy, rx, ry, ang=0):
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (cx, cy), (rx, ry), ang, 0, 360, 255, -1)
    return m.astype(bool)


def rect_mask(x, y, w, h):
    m = np.zeros((H, W), np.uint8)
    cv2.rectangle(m, (x, y), (x + w, y + h), 255, -1)
    return m.astype(bool)


# --- 1. deforestation / clearing patches -> bare tan soil
clear = (ellipse_mask(int(W * .78), int(H * .30), 78, 58, 20) |
         ellipse_mask(int(W * .70), int(H * .22), 52, 40, -10) |
         ellipse_mask(int(W * .30), int(H * .62), 64, 46, 35))
paste(clear, (168, 146, 112), "vegetation_cleared_to_bare", 16)

# --- 2. urban expansion block -> grey impervious with a road grid
urb = rect_mask(int(W * .13), int(H * .78), 165, 120)
paste(urb, (126, 126, 130), "vegetation_to_builtup", 22)
for gx in range(int(W * .13), int(W * .13) + 165, 28):      # street grid texture
    cv2.line(t2, (gx, int(H * .78)), (gx, int(H * .78) + 120), (150, 150, 152), 2)
for gy in range(int(H * .78), int(H * .78) + 120, 26):
    cv2.line(t2, (int(W * .13), gy), (int(W * .13) + 165, gy), (150, 150, 152), 2)

# --- 3. new reservoir / inundation -> dark blue water
water = ellipse_mask(int(W * .52), int(H * .80), 92, 52, 12)
paste(water, (38, 62, 96), "land_to_water", 9)

# --- 4. geometric offset so registration is exercised
M = np.float32([[1, 0, 6], [0, 1, -4]])
t2 = cv2.warpAffine(t2, M, (W, H), borderMode=cv2.BORDER_REPLICATE)

# --- 5. mild illumination / atmospheric difference between dates
t2 = np.clip(t2.astype(np.float32) * 1.06 + 5, 0, 255).astype(np.uint8)

Image.fromarray(t1).save("samples/delta_t1.png")
Image.fromarray(t2).save("samples/delta_t2.png")

total_changed = sum(t["fraction"] for t in truth)
truth_doc = {"source": SRC, "shape": [H, W], "edits": truth,
             "total_changed_fraction": round(total_changed, 5),
             "applied_shift_px": [6, -4],
             "note": "Ground truth for validating change detection output."}
with open("samples/delta_truth.json", "w") as f:
    json.dump(truth_doc, f, indent=2)
print(json.dumps(truth_doc, indent=2))
