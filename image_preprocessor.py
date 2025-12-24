import cv2
import numpy as np
import os
import math

class ImagePreprocessor:
    def __init__(self, debug=False, debug_dir="preprocess_debug"):
        self.TARGET_RATIO = 1.58
        self.MIN_AREA_RATIO = 0.05
        
        self.debug = debug
        self.debug_dir = debug_dir
        
        if self.debug:
            os.makedirs(self.debug_dir, exist_ok=True)

        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def preprocess(self, image):
        self._save(image, "00_original")

        oriented_image = self.correct_orientation_semantic(image)
        self._save(oriented_image, "01_orientation_fixed")

        warped, was_warped = self.geometric_correction(oriented_image)
        
        name = "02_warped" if was_warped else "02_warp_skipped"
        self._save(warped, name)

        deskewed = self.deskew_hough(warped)
        self._save(deskewed, "03_deskewed")
        
        final_image = self.add_padding(deskewed)
        self._save(final_image, "04_final_padded")

        self._save_comparison(image, final_image)
        
        return final_image

    def add_padding(self, image, pad_size=20):
        """Adds a white border around the image to help OCR with edge characters."""
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
            if angle == 0: rotated = small
            else: rotated = self.rotate_image_90(small, angle)

            gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            if len(faces) > max_faces:
                max_faces = len(faces)
                best_angle = angle
        
        if max_faces == 0:
            if h > w: return self.rotate_image_90(image, 90)
            return image

        return self.rotate_image_90(image, best_angle)

    def rotate_image_90(self, image, angle):
        if angle == 0: return image
        if angle == 90: return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if angle == 180: return cv2.rotate(image, cv2.ROTATE_180)
        if angle == 270: return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image

    def geometric_correction(self, image):
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 30, 100) 
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edged, kernel, iterations=2)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
                    if self.debug: print("DEBUG: Contour is full image. Skipping warp.")
                    return image, False

                if not self.should_warp(pts, w, h):
                    if self.debug: print("DEBUG: Shape is flat (not trapezoid). Skipping warp.")
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

        IS_TRAPEZOID = (width_ratio < 0.85) or (height_ratio < 0.85)

        if not IS_TRAPEZOID:
            return False 
            
        avg_width = (top_width + bottom_width) / 2
        avg_height = (left_height + right_height) / 2
        if avg_height == 0: return False
        ar = avg_width / avg_height
        
        if not (1.2 < ar < 2.0):
            return False 

        return True

    def deskew_hough(self, image):
        (h, w) = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        margin_x = int(w * 0.2)
        margin_y = int(h * 0.2)
        roi = gray[margin_y:h-margin_y, margin_x:w-margin_x]
        
        if roi.size == 0: return image

        thresh = cv2.adaptiveThreshold(
            roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 31, 15
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        
        if self.debug: self._save(dilated, "debug_hough_lines_roi")

        lines = cv2.HoughLinesP(
            dilated, rho=1, theta=np.pi/180, threshold=50, 
            minLineLength=roi.shape[1] // 4, 
            maxLineGap=20
        )

        if lines is None: return image

        angles = []
        weights = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            if -15 < angle < 15:
                angles.append(angle)
                weights.append(length)

        if not angles: return image

        avg_angle = np.average(angles, weights=weights)
        
        if self.debug: print(f"DEBUG: Hough Weighted Angle: {avg_angle}")

        if abs(avg_angle) < 0.5:
            return image

        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
        
        rotated = cv2.warpAffine(
            image, M, (w, h), 
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        return rotated

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        rect[0] = pts[np.argmin(s)]      
        rect[2] = pts[np.argmax(s)]      
        rect[1] = pts[np.argmin(diff)]   
        rect[3] = pts[np.argmax(diff)]   
        return rect

    def four_point_transform(self, image, pts):
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))
        dst = np.array([
            [0, 0], [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    def _save(self, img, name):
        if not self.debug or img is None: return
        import time
        ts = int(time.time() * 1000)
        cv2.imwrite(os.path.join(self.debug_dir, f"{ts}_{name}.jpg"), img)

    def _save_comparison(self, original, processed):
        if not self.debug or original is None or processed is None: return
        h = max(original.shape[0], processed.shape[0])
        scale = h / processed.shape[0]
        p_resized = cv2.resize(processed, None, fx=scale, fy=scale)
        scale_o = h / original.shape[0]
        o_resized = cv2.resize(original, None, fx=scale_o, fy=scale_o)
        canvas = np.concatenate((o_resized, p_resized), axis=1)
        self._save(canvas, "comparison")