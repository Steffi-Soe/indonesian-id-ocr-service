import cv2
import numpy as np
import os
import math


class StandardPreprocessor:
    def __init__(self, debug=False, debug_dir="preprocess_debug"):
        self.TARGET_RATIO = 1.58
        self.MIN_AREA_RATIO = 0.05
        self.PROCESSING_WIDTH = 1280
        self.OUTPUT_WIDTH = 1000
        self.debug = debug
        self.debug_dir = debug_dir

        if self.debug:
            os.makedirs(self.debug_dir, exist_ok=True)

        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def resize_keep_aspect(self, image, target_width):
        h, w = image.shape[:2]
        if w == target_width:
            return image
        scale = target_width / w
        return cv2.resize(image, None, fx=scale, fy=scale)

    def preprocess(self, image):
        h, w = image.shape[:2]
        if w > self.PROCESSING_WIDTH:
            image = self.resize_keep_aspect(image, self.PROCESSING_WIDTH)

        oriented_image = self.correct_orientation_semantic(image)
        if self.debug:
            self._save(oriented_image, "std_01_oriented")

        warped, _ = self.geometric_correction(oriented_image)
        if self.debug:
            self._save(warped, "std_02_warped")

        deskewed = self.deskew_hough(warped)
        if self.debug:
            self._save(deskewed, "std_03_deskewed")

        final_normalized = self.resize_keep_aspect(
            deskewed,
            self.OUTPUT_WIDTH
        )
        final_image = self.add_padding(final_normalized)
        if self.debug:
            self._save(final_image, "std_04_final")

        return final_image

    def add_padding(self, image, pad_size=20):
        return cv2.copyMakeBorder(
            image,
            pad_size, pad_size, pad_size, pad_size,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        )

    def correct_orientation_semantic(self, image):
        h, w = image.shape[:2]
        scale = 600 / max(h, w)
        small = cv2.resize(image, None, fx=scale, fy=scale)

        best_angle = 0
        max_faces = 0

        for angle in [0, 90, 180, 270]:
            if angle == 0:
                rotated = small
            else:
                rotated = self.rotate_image_90(small, angle)

            gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            if len(faces) > max_faces:
                max_faces = len(faces)
                best_angle = angle

        if max_faces == 0:
            if h > w:
                return self.rotate_image_90(image, 90)
            return image

        return self.rotate_image_90(image, best_angle)

    def rotate_image_90(self, image, angle):
        if angle == 0:
            return image
        if angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        if angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image

    def geometric_correction(self, image):
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 30, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edged, kernel, iterations=2)

        contours, _ = cv2.findContours(
            dilated,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for c in contours:
            area = cv2.contourArea(c)
            if area < (h * w * self.MIN_AREA_RATIO):
                continue

            peri = cv2.arcLength(c, True)
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

    def should_warp(self, pts, img_w, img_h):
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        top_width = np.linalg.norm(tr - tl)
        bottom_width = np.linalg.norm(br - bl)
        left_height = np.linalg.norm(bl - tl)
        right_height = np.linalg.norm(br - tr)

        width_ratio = min(top_width, bottom_width) / max(top_width, bottom_width)
        height_ratio = min(left_height, right_height) / max(left_height, right_height)

        if (width_ratio < 0.85) or (height_ratio < 0.85):
            avg_width = (top_width + bottom_width) / 2
            avg_height = (left_height + right_height) / 2
            if avg_height == 0:
                return False
            ar = avg_width / avg_height
            if 1.2 < ar < 2.0:
                return True
        return False

    def deskew_hough(self, image):
        (h, w) = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        margin_x = int(w * 0.2)
        margin_y = int(h * 0.2)
        roi = gray[margin_y:h - margin_y, margin_x:w - margin_x]
        if roi.size == 0:
            return image

        thresh = cv2.adaptiveThreshold(
            roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 15
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        dilated = cv2.dilate(thresh, kernel, iterations=1)

        lines = cv2.HoughLinesP(
            dilated, rho=1, theta=np.pi / 180, threshold=50,
            minLineLength=roi.shape[1] // 4, maxLineGap=20
        )
        if lines is None:
            return image

        angles = []
        weights = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if -15 < angle < 15:
                angles.append(angle)
                weights.append(length)

        if not angles:
            return image
        avg_angle = np.average(angles, weights=weights)

        if abs(avg_angle) < 0.5:
            return image
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
        return cv2.warpAffine(
            image,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def four_point_transform(self, image, pts):
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
        max_width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        max_height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (max_width, max_height))

    def _save(self, img, name):
        if not self.debug or img is None:
            return
        import time
        ts = int(time.time() * 1000)
        cv2.imwrite(os.path.join(self.debug_dir, f"{ts}_{name}.jpg"), img)


class SmartSIMPreprocessor(StandardPreprocessor):
    def __init__(self, debug=False, debug_dir="preprocess_debug"):
        super().__init__(debug, debug_dir)
        self.OUTPUT_WIDTH = 1600
        self.PROCESSING_WIDTH = 1280

    def preprocess(self, image):
        oriented_image = self.correct_orientation_semantic(image)
        if self.debug:
            self._save(oriented_image, "smart_01_oriented")

        warped, _ = self.geometric_correction_high_res(oriented_image)
        if self.debug:
            self._save(warped, "smart_02_warped")

        deskewed = self.deskew_hough_high_res(warped)
        if self.debug:
            self._save(deskewed, "smart_03_deskewed")

        enhanced = self.enhance_details(deskewed)
        if self.debug:
            self._save(enhanced, "smart_04_enhanced")

        h, w = enhanced.shape[:2]
        if w > self.OUTPUT_WIDTH:
            final_normalized = self.resize_keep_aspect(
                enhanced,
                self.OUTPUT_WIDTH
            )
        else:
            final_normalized = enhanced

        final_image = self.add_padding(final_normalized)
        if self.debug:
            self._save(final_image, "smart_05_final")

        return final_image

    def geometric_correction_high_res(self, full_image):
        h, w = full_image.shape[:2]
        detect_img = self.resize_keep_aspect(
            full_image,
            self.PROCESSING_WIDTH
        )
        scale = w / detect_img.shape[1]

        gray = cv2.cvtColor(detect_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 30, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edged, kernel, iterations=2)

        contours, _ = cv2.findContours(
            dilated,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for c in contours:
            area = cv2.contourArea(c)
            min_area = detect_img.shape[0] * detect_img.shape[1] * self.MIN_AREA_RATIO
            if area < min_area:
                continue

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4:
                pts = approx.reshape(4, 2)
                x, y, cw, ch = cv2.boundingRect(c)
                if cw > 0.95 * detect_img.shape[1] and ch > 0.95 * detect_img.shape[0]:
                    continue

                if self.should_warp(pts, detect_img.shape[1], detect_img.shape[0]):
                    full_pts = pts.astype(np.float32) * scale
                    warped = self.four_point_transform(full_image, full_pts)
                    return warped, True

        return full_image, False

    def deskew_hough_high_res(self, image):
        h_orig, w_orig = image.shape[:2]
        detect_img = self.resize_keep_aspect(image, 1000)
        h, w = detect_img.shape[:2]

        gray = cv2.cvtColor(detect_img, cv2.COLOR_BGR2GRAY)
        margin_x, margin_y = int(w * 0.15), int(h * 0.15)
        roi = gray[margin_y:h - margin_y, margin_x:w - margin_x]

        if roi.size == 0:
            return image

        thresh = cv2.adaptiveThreshold(
            roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 15
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        dilated = cv2.dilate(thresh, kernel, iterations=1)

        lines = cv2.HoughLinesP(
            dilated,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=w // 4,
            maxLineGap=20
        )
        if lines is None:
            return image

        angles = []
        weights = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if -20 < angle < 20:
                angles.append(angle)
                weights.append(length)

        if not angles:
            return image
        avg_angle = np.average(angles, weights=weights)

        if abs(avg_angle) < 0.5:
            return image
        center = (w_orig // 2, h_orig // 2)
        M = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
        return cv2.warpAffine(
            image,
            M,
            (w_orig, h_orig),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

    def enhance_details(self, image):
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            return cv2.fastNlMeansDenoisingColored(
                enhanced, None, 3, 3, 7, 21
            )
        except Exception:
            return image