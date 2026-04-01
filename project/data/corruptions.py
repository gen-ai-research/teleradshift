import io
import numpy as np
import cv2
from PIL import Image

def to_uint8(img):
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def pil_to_np(pil_img):
    img = np.array(pil_img)
    # Ensure channel dimension exists
    if img.ndim == 2:           # (H, W)
        img = img[:, :, None]   # (H, W, 1)
    return img

def np_to_pil(img):
    img = to_uint8(img)
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[:, :, 0]
    return Image.fromarray(img)

def jpeg_compress(pil_img, quality):
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert(pil_img.mode)

def gaussian_blur(pil_img, sigma):
    img = pil_to_np(pil_img)
    k = int(2 * round(3 * sigma) + 1)
    blurred = cv2.GaussianBlur(img, (k, k), sigmaX=sigma)
    return np_to_pil(blurred)

def brightness_shift(pil_img, factor):
    img = pil_to_np(pil_img).astype(np.float32)
    img = img * float(factor)
    return np_to_pil(img)

def occlusion(pil_img, frac=0.1, seed=0):
    rng = np.random.default_rng(seed)
    img = pil_to_np(pil_img).copy()
    h, w = img.shape[:2]
    area = int(frac * h * w)
    side = int(np.sqrt(area))
    side = max(10, side)
    x = rng.integers(0, max(1, w - side))
    y = rng.integers(0, max(1, h - side))
    img[y:y+side, x:x+side, :] = 0
    return np_to_pil(img)

def photo_of_screen(pil_img, severity=1, seed=0):
    """
    severity: 1 (mild), 2 (medium), 3 (strong)
    """
    rng = np.random.default_rng(seed)
    img = pil_to_np(pil_img).astype(np.float32)
    h, w = img.shape[:2]

    # 1) Perspective warp
    margin = {1: 0.03, 2: 0.06, 3: 0.10}[severity]
    dx = int(margin * w)
    dy = int(margin * h)

    src = np.float32([[0,0],[w,0],[w,h],[0,h]])
    dst = np.float32([
        [rng.integers(0, dx), rng.integers(0, dy)],
        [w - rng.integers(0, dx), rng.integers(0, dy)],
        [w - rng.integers(0, dx), h - rng.integers(0, dy)],
        [rng.integers(0, dx), h - rng.integers(0, dy)]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # 2) Moiré pattern overlay (sinusoidal grid)
    freq = {1: 8, 2: 5, 3: 3}[severity]  # lower -> stronger visible stripes
    amp  = {1: 6, 2: 12, 3: 20}[severity]

    yy, xx = np.mgrid[0:h, 0:w]
    moire = (np.sin(2*np.pi*xx/(w/freq)) + np.sin(2*np.pi*yy/(h/freq))) / 2.0
    moire = moire[:, :, None] * amp
    warped = warped + moire

    # 3) Glare spot (Gaussian blob)
    glare_strength = {1: 0.15, 2: 0.25, 3: 0.35}[severity]
    gx = rng.integers(int(0.2*w), int(0.8*w))
    gy = rng.integers(int(0.2*h), int(0.8*h))
    sigma = {1: 0.10, 2: 0.18, 3: 0.25}[severity] * min(h, w)

    glare = np.exp(-(((xx-gx)**2 + (yy-gy)**2) / (2*sigma**2)))
    glare = glare[:, :, None] * (255.0 * glare_strength)
    warped = warped + glare

    return np_to_pil(warped)

def apply_corruption(pil_img, corr_name, severity, seed=0):
    if corr_name == "jpeg":
        q = {1: 40, 2: 20, 3: 10}[severity]
        return jpeg_compress(pil_img, q)
    if corr_name == "blur":
        s = {1: 1, 2: 3, 3: 5}[severity]
        return gaussian_blur(pil_img, s)
    if corr_name == "brightness":
        f = {1: 0.85, 2: 0.65, 3: 0.45}[severity]
        return brightness_shift(pil_img, f)
    if corr_name == "occlusion":
        frac = {1: 0.10, 2: 0.20, 3: 0.30}[severity]
        return occlusion(pil_img, frac=frac, seed=seed)
    if corr_name == "screen":
        return photo_of_screen(pil_img, severity=severity, seed=seed)
    raise ValueError("Unknown corruption")