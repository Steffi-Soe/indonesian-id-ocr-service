import cv2
import numpy as np
import os
import math


# ---------------------------------------------------------------------------
# Quality Assessment
# ---------------------------------------------------------------------------

class ImageQualityAssessor:
    """Measures image quality to drive adaptive preprocessing decisions."""

    @staticmethod
    def blur_score(image: np.ndarray) -> float:
        """Laplacian variance — higher = sharper. Blurry images score < 80."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def brightness(image: np.ndarray) -> float:
        """Mean pixel brightness [0-255]."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return float(np.mean(gray))

    @staticmethod
    def contrast(image: np.ndarray) -> float:
        """Standard deviation of brightness [0-128]."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return float(np.std(gray))

    @classmethod
    def assess(cls, image: np.ndarray) -> dict:
        blur   = cls.blur_score(image)
        bright = cls.brightness(image)
        cont   = cls.contrast(image)
        return {
            "blur":            blur,
            "brightness":      bright,
            "contrast":        cont,
            "is_blurry":       blur   < 80,
            "is_very_blurry":  blur   < 30,
            "is_dark":         bright < 60,
            "is_overexposed":  bright > 210,
            "is_low_contrast": cont   < 30,
        }


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def unsharp_mask(image: np.ndarray, sigma: float = 1.0, strength: float = 1.5) -> np.ndarray:
    """Sharpen an image via unsharp masking."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def normalize_exposure(image: np.ndarray, clip_limit: float = 3.0) -> np.ndarray:
    """Apply CLAHE to the L channel of LAB to normalise exposure."""
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    except Exception:
        return image


# ---------------------------------------------------------------------------
# StandardPreprocessor  (used for KTP and initial SIM pass)
# ---------------------------------------------------------------------------

class StandardPreprocessor:
    def __init__(self, debug=False, debug_dir="preprocess_debug"):
        self.TARGET_RATIO  = 1.58
        self.MIN_AREA_RATIO = 0.05
        self.PROCESSING_WIDTH = 1280
        self.OUTPUT_WIDTH     = 1000
        self.debug     = debug
        self.debug_dir = debug_dir

        if self.debug:
            os.makedirs(self.debug_dir, exist_ok=True)

        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade    = cv2.CascadeClassifier(cascade_path)
        self.quality_assessor = ImageQualityAssessor()

    # ------------------------------------------------------------------
    def resize_keep_aspect(self, image, target_width):
        h, w = image.shape[:2]
        if w == target_width:
            return image
        scale = target_width / w
        interp = cv2.INTER_LANCZOS4 if scale < 1.0 else cv2.INTER_LINEAR
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=interp)

    # ------------------------------------------------------------------
    def preprocess(self, image):
        h, w = image.shape[:2]
        if w > self.PROCESSING_WIDTH:
            image = self.resize_keep_aspect(image, self.PROCESSING_WIDTH)

        quality = self.quality_assessor.assess(image)

        # Pre-enhance dark / low-contrast images before other steps
        if quality["is_dark"] or quality["is_low_contrast"]:
            clip = 4.5 if quality["is_dark"] else 3.0
            image = normalize_exposure(image, clip_limit=clip)
            if self.debug:
                self._save(image, "std_00_exposure_fixed")

        oriented_image = self.correct_orientation_semantic(image)
        if self.debug:
            self._save(oriented_image, "std_01_oriented")

        warped, _ = self.geometric_correction(oriented_image)
        if self.debug:
            self._save(warped, "std_02_warped")

        deskewed = self.deskew_hough(warped)
        if self.debug:
            self._save(deskewed, "std_03_deskewed")

        final_normalized = self.resize_keep_aspect(deskewed, self.OUTPUT_WIDTH)

        # Post-resize sharpening if still blurry
        post_q = self.quality_assessor.assess(final_normalized)
        if post_q["is_blurry"]:
            strength = 1.8 if post_q["is_very_blurry"] else 1.2
            final_normalized = unsharp_mask(final_normalized, sigma=0.8, strength=strength)

        final_image = self.add_padding(final_normalized)
        if self.debug:
            self._save(final_image, "std_04_final")

        return final_image

    # ------------------------------------------------------------------
    def add_padding(self, image, pad_size=20):
        return cv2.copyMakeBorder(
            image,
            pad_size, pad_size, pad_size, pad_size,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        )

    # ------------------------------------------------------------------
    def minimal_preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Non-destructive KTP preprocessing path (v3 — authoritative).

        Operations applied (in order):
          1. Orientation correction via face detection (portrait → landscape).
          2. Resize to OUTPUT_WIDTH (1000 px), preserving aspect ratio.
          3. Add 20-px white border padding to prevent OCR edge-clipping.

        Deliberately omits:
          - Deskewing / skew correction
          - Geometric / perspective correction
          - Sharpening / unsharp masking
          - CLAHE or any contrast manipulation
          - Denoising
          - Any morphological operation

        KTP images arriving at the API are assumed to be high quality.
        Preserving the original pixel data maximises OCR accuracy.
        """
        oriented = self.correct_orientation_semantic(image)
        resized  = self.resize_keep_aspect(oriented, self.OUTPUT_WIDTH)
        return self.add_padding(resized)

    # ------------------------------------------------------------------
    def correct_orientation_semantic(self, image):
        h, w = image.shape[:2]
        scale = 600 / max(h, w)
        small = cv2.resize(image, None, fx=scale, fy=scale)

        best_angle = 0
        max_faces  = 0

        for angle in [0, 90, 180, 270]:
            rotated = small if angle == 0 else self.rotate_image_90(small, angle)
            gray    = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
            faces   = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            if len(faces) > max_faces:
                max_faces = len(faces)
                best_angle = angle

        if max_faces == 0:
            # Fallback: identity cards are landscape — rotate portrait images
            if h > w:
                return self.rotate_image_90(image, 90)
            return image

        return self.rotate_image_90(image, best_angle)

    # ------------------------------------------------------------------
    def rotate_image_90(self, image, angle):
        if angle == 0:   return image
        if angle == 90:  return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if angle == 180: return cv2.rotate(image, cv2.ROTATE_180)
        if angle == 270: return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image

    # ------------------------------------------------------------------
    def geometric_correction(self, image):
        h, w   = image.shape[:2]
        gray   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged  = cv2.Canny(blurred, 30, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edged, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for c in contours:
            area = cv2.contourArea(c)
            if area < (h * w * self.MIN_AREA_RATIO):
                continue

            peri  = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4:
                pts = approx.reshape(4, 2)
                x, y, cw, ch = cv2.boundingRect(c)
                if cw > 0.95 * w and ch > 0.95 * h:
                    return image, False
                if not self.should_warp(pts, w, h):
                    return image, False

                warped = self.four_point_transform(image, pts)
                return warped, True

        return image, False

    # ------------------------------------------------------------------
    def should_warp(self, pts, img_w, img_h):
        rect = self.order_points(pts)
        tl, tr, br, bl = rect

        top_w    = np.linalg.norm(tr - tl)
        bot_w    = np.linalg.norm(br - bl)
        left_h   = np.linalg.norm(bl - tl)
        right_h  = np.linalg.norm(br - tr)

        w_ratio = min(top_w, bot_w) / (max(top_w, bot_w) + 1e-6)
        h_ratio = min(left_h, right_h) / (max(left_h, right_h) + 1e-6)

        if w_ratio < 0.85 or h_ratio < 0.85:
            avg_w = (top_w + bot_w) / 2
            avg_h = (left_h + right_h) / 2
            if avg_h == 0:
                return False
            ar = avg_w / avg_h
            if 1.2 < ar < 2.0:
                return True
        return False

    # ------------------------------------------------------------------
    def deskew_hough(self, image):
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        mx, my = int(w * 0.2), int(h * 0.2)
        roi = gray[my:h - my, mx:w - mx]
        if roi.size == 0:
            return image

        thresh  = cv2.adaptiveThreshold(
            roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
        )
        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        dilated = cv2.dilate(thresh, kernel, iterations=1)

        lines = cv2.HoughLinesP(
            dilated, 1, np.pi / 180, 50,
            minLineLength=roi.shape[1] // 4, maxLineGap=20
        )
        if lines is None:
            return image

        angles, weights = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle  = math.degrees(math.atan2(y2 - y1, x2 - x1))
            length = math.hypot(x2 - x1, y2 - y1)
            if -15 < angle < 15:
                angles.append(angle)
                weights.append(length)

        if not angles:
            return image
        avg_angle = np.average(angles, weights=weights)

        if abs(avg_angle) < 0.5:
            return image
        M = cv2.getRotationMatrix2D((w // 2, h // 2), avg_angle, 1.0)
        return cv2.warpAffine(image, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    # ------------------------------------------------------------------
    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s    = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff    = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def four_point_transform(self, image, pts):
        rect = self.order_points(pts)
        tl, tr, br, bl = rect
        max_w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        max_h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
        dst   = np.array([[0, 0], [max_w - 1, 0],
                           [max_w - 1, max_h - 1], [0, max_h - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (max_w, max_h))

    # ------------------------------------------------------------------
    def _save(self, img, name):
        if not self.debug or img is None:
            return
        import time
        ts = int(time.time() * 1000)
        cv2.imwrite(os.path.join(self.debug_dir, f"{ts}_{name}.jpg"), img)


# ---------------------------------------------------------------------------
# SmartSIMPreprocessor  (high-res path for newer SIM layouts)
# ---------------------------------------------------------------------------

class SmartSIMPreprocessor(StandardPreprocessor):
    def __init__(self, debug=False, debug_dir="preprocess_debug"):
        super().__init__(debug, debug_dir)
        self.OUTPUT_WIDTH     = 1600
        self.PROCESSING_WIDTH = 1280

    # ------------------------------------------------------------------
    def preprocess(self, image):
        quality = self.quality_assessor.assess(image)

        oriented_image = self.correct_orientation_semantic(image)
        if self.debug:
            self._save(oriented_image, "smart_01_oriented")

        warped, _ = self.geometric_correction_high_res(oriented_image)
        if self.debug:
            self._save(warped, "smart_02_warped")

        deskewed = self.deskew_hough_high_res(warped)
        if self.debug:
            self._save(deskewed, "smart_03_deskewed")

        enhanced = self._enhance_details(deskewed, quality)
        if self.debug:
            self._save(enhanced, "smart_04_enhanced")

        h, w = enhanced.shape[:2]
        final_normalized = (self.resize_keep_aspect(enhanced, self.OUTPUT_WIDTH)
                            if w > self.OUTPUT_WIDTH else enhanced)

        final_image = self.add_padding(final_normalized)
        if self.debug:
            self._save(final_image, "smart_05_final")

        return final_image

    # ------------------------------------------------------------------
    def geometric_correction_high_res(self, full_image):
        h, w        = full_image.shape[:2]
        detect_img  = self.resize_keep_aspect(full_image, self.PROCESSING_WIDTH)
        scale       = w / detect_img.shape[1]

        gray    = cv2.cvtColor(detect_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged   = cv2.Canny(blurred, 30, 100)
        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edged, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for c in contours:
            area     = cv2.contourArea(c)
            min_area = detect_img.shape[0] * detect_img.shape[1] * self.MIN_AREA_RATIO
            if area < min_area:
                continue

            peri   = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4:
                pts  = approx.reshape(4, 2)
                x, y, cw, ch = cv2.boundingRect(c)
                if cw > 0.95 * detect_img.shape[1] and ch > 0.95 * detect_img.shape[0]:
                    continue
                if self.should_warp(pts, detect_img.shape[1], detect_img.shape[0]):
                    full_pts = pts.astype(np.float32) * scale
                    warped   = self.four_point_transform(full_image, full_pts)
                    return warped, True

        return full_image, False

    # ------------------------------------------------------------------
    def deskew_hough_high_res(self, image):
        h_orig, w_orig = image.shape[:2]
        detect_img     = self.resize_keep_aspect(image, 1000)
        h, w           = detect_img.shape[:2]

        gray  = cv2.cvtColor(detect_img, cv2.COLOR_BGR2GRAY)
        mx, my = int(w * 0.15), int(h * 0.15)
        roi   = gray[my:h - my, mx:w - mx]
        if roi.size == 0:
            return image

        thresh  = cv2.adaptiveThreshold(
            roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
        )
        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        dilated = cv2.dilate(thresh, kernel, iterations=1)

        lines = cv2.HoughLinesP(
            dilated, 1, np.pi / 180, 50,
            minLineLength=w // 4, maxLineGap=20
        )
        if lines is None:
            return image

        angles, weights = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            angle  = math.degrees(math.atan2(y2 - y1, x2 - x1))
            length = math.hypot(x2 - x1, y2 - y1)
            if -20 < angle < 20:
                angles.append(angle)
                weights.append(length)

        if not angles:
            return image
        avg_angle = np.average(angles, weights=weights)
        if abs(avg_angle) < 0.5:
            return image

        M = cv2.getRotationMatrix2D((w_orig // 2, h_orig // 2), avg_angle, 1.0)
        return cv2.warpAffine(image, M, (w_orig, h_orig),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    # ------------------------------------------------------------------
    def _enhance_details(self, image, quality=None):
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # Adaptive CLAHE clip based on image quality
            clip = 2.0
            if quality:
                if quality.get("is_dark"):          clip = 4.5
                elif quality.get("is_low_contrast"): clip = 4.0
                elif quality.get("is_blurry"):       clip = 3.0

            clahe     = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
            cl        = clahe.apply(l)
            enhanced  = cv2.cvtColor(cv2.merge([cl, a, b]), cv2.COLOR_LAB2BGR)
            denoised  = cv2.fastNlMeansDenoisingColored(enhanced, None, 3, 3, 7, 21)

            # Apply sharpening on blurry images
            if quality and quality.get("is_blurry"):
                strength = 2.0 if quality.get("is_very_blurry") else 1.5
                denoised = unsharp_mask(denoised, sigma=0.8, strength=strength)

            return denoised
        except Exception:
            return image