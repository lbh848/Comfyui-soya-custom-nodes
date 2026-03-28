import torch
import numpy as np
import cv2
import os
import urllib.request


class AnimeBlinkDetector_mdsoya:
    """
    애니메이션 캐릭터 눈 깜빡임 감지 노드.
    lbpcascade_animeface + 암색 픽셀 비율 기반.

    회전 보정:
      1. 4방향(0°/90°/180°/270°) 회전 후 예상 눈 위치의 암색 점수 비교
      2. 최적 방향 선택 후 미세 기울기(틸트) 보정
      3. 보정된 이미지로 blink 판별 + 보정 이미지 출력
    """

    CASCADE_URL = (
        "https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/"
        "master/lbpcascade_animeface.xml"
    )

    def __init__(self):
        cascade_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "lbpcascade_animeface.xml"
        )
        if not os.path.exists(cascade_path):
            print("[Soya: AnimeBlink] lbpcascade_animeface.xml 다운로드 중...")
            urllib.request.urlretrieve(self.CASCADE_URL, cascade_path)
        self.cascade = cv2.CascadeClassifier(cascade_path)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": (
                    "FLOAT",
                    {"default": 0.15, "min": 0.01, "max": 0.80, "step": 0.01},
                ),
                "is_face_crop": (
                    "BOOLEAN",
                    {"default": True},
                ),
            },
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "IMAGE")
    RETURN_NAMES = ("is_closed", "openness", "corrected_image")
    OUTPUT_IS_LIST = (True, True, False)
    FUNCTION = "detect_blink"
    CATEGORY = "Soya"

    # ------------------------------------------------------------------ #
    #  회전 보정
    # ------------------------------------------------------------------ #

    @staticmethod
    def _rotate90(image, angle):
        if angle == 0:
            return image
        if angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        if angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image

    @staticmethod
    def _dark_score(roi):
        """ROI의 암색 픽셀 비율 (Otsu 이진화)."""
        if roi.size == 0 or roi.shape[0] < 4 or roi.shape[1] < 4:
            return -1.0
        blur = cv2.GaussianBlur(roi, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return float(np.sum(binary > 0)) / binary.size

    def _score_eye_regions(self, gray):
        """정면 얼굴 기준 예상 눈 위치의 암색 점수. 높을수록 곧은 얼굴."""
        h, w = gray.shape
        if h < 20 or w < 20:
            return -1.0

        le = gray[int(h * 0.25):int(h * 0.55), int(w * 0.08):int(w * 0.44)]
        re = gray[int(h * 0.25):int(h * 0.55), int(w * 0.56):int(w * 0.92)]

        ls = self._dark_score(le)
        rs = self._dark_score(re)
        if ls < 0 or rs < 0:
            return -1.0

        symmetry = 1.0 - abs(ls - rs)
        return (ls + rs) * symmetry

    def _compute_rotation(self, gray):
        """최적 회전 파라미터 (rot90_angle, tilt_angle) 계산."""
        best_angle = 0
        best_score = -1.0

        for angle in [0, 90, 180, 270]:
            rotated = self._rotate90(gray, angle)
            score = self._score_eye_regions(rotated)
            if score > best_score:
                best_score = score
                best_angle = angle

        tilt = self._compute_tilt(self._rotate90(gray, best_angle))
        return best_angle, tilt

    @staticmethod
    def _darkest_center(roi):
        """ROI 내 가장 큰 암색 영역의 중심 좌표 반환."""
        if roi.size == 0:
            return None
        blur = cv2.GaussianBlur(roi, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None

        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    def _compute_tilt(self, gray):
        """대략 곧은 얼굴에서 미세 기울기 각도 계산."""
        h, w = gray.shape

        le_roi = gray[int(h * 0.25):int(h * 0.55), int(w * 0.08):int(w * 0.44)]
        re_roi = gray[int(h * 0.25):int(h * 0.55), int(w * 0.56):int(w * 0.92)]

        le_c = self._darkest_center(le_roi)
        re_c = self._darkest_center(re_roi)

        if le_c is None or re_c is None:
            return 0.0

        le_x = le_c[0] + int(w * 0.08)
        le_y = le_c[1] + int(h * 0.25)
        re_x = re_c[0] + int(w * 0.56)
        re_y = re_c[1] + int(h * 0.25)

        angle = np.degrees(np.arctan2(re_y - le_y, re_x - le_x))

        if abs(angle) > 45.0 or abs(angle) < 1.0:
            return 0.0
        return angle

    @staticmethod
    def _apply_tilt(image, angle):
        """이미지에 미세 기울기 보정 적용 (gray/rgb 모두 가능)."""
        if abs(angle) < 1.0:
            return image
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        border = 255 if image.ndim == 2 else (255, 255, 255)
        return cv2.warpAffine(image, M, (w, h), borderValue=border)

    def _apply_rotation(self, image, rot90, tilt):
        """90° 회전 + 미세 틸트를 이미지에 적용."""
        result = self._rotate90(image, rot90)
        result = self._apply_tilt(result, tilt)
        return result

    # ------------------------------------------------------------------ #
    #  눈 분석
    # ------------------------------------------------------------------ #

    @staticmethod
    def _eye_rois(face, img_h, img_w):
        """얼굴 bbox → 좌/우 눈 ROI (x1, y1, x2, y2)."""
        fx, fy, fw, fh = face

        ey1 = max(int(fy + fh * 0.25), 0)
        ey2 = min(int(fy + fh * 0.55), img_h)

        lx1 = max(int(fx + fw * 0.08), 0)
        lx2 = min(int(fx + fw * 0.44), img_w)
        rx1 = max(int(fx + fw * 0.56), 0)
        rx2 = min(int(fx + fw * 0.92), img_w)

        return (lx1, ey1, lx2, ey2), (rx1, ey1, rx2, ey2)

    @staticmethod
    def _calc_openness(gray, x1, y1, x2, y2):
        """눈 영역 열림 정도(0~1). 상단 20% 제외 → Otsu → 암색 비율."""
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        h, w = roi.shape
        if h < 6 or w < 6:
            return 0.0

        iris_zone = roi[int(h * 0.2):, :]
        blur = cv2.GaussianBlur(iris_zone, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        return float(np.sum(binary > 0)) / binary.size

    # ------------------------------------------------------------------ #
    #  메인 실행
    # ------------------------------------------------------------------ #

    def detect_blink(self, image, threshold, is_face_crop):
        is_closed_list = []
        openness_list = []
        corrected_imgs = []

        for i, img in enumerate(image):
            img_np = (255.0 * img.cpu().numpy()).astype(np.uint8)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

            if is_face_crop:
                rot90, tilt = self._compute_rotation(gray)

                # 동일한 변환을 gray와 RGB 모두에 적용
                corrected_gray = self._apply_rotation(gray, rot90, tilt)
                corrected_rgb = self._apply_rotation(img_np, rot90, tilt)
                face = (0, 0, corrected_gray.shape[1], corrected_gray.shape[0])

                if abs(rot90) > 0.5 or abs(tilt) > 0.5:
                    print(
                        f"[Soya: AnimeBlink] Batch {i} 회전 보정: "
                        f"{rot90}° + tilt {tilt:.1f}°"
                    )
            else:
                # cascade로 4방향 얼굴 검출
                found = False
                for angle in [0, 90, 180, 270]:
                    rotated_gray = self._rotate90(gray, angle)
                    faces = self.cascade.detectMultiScale(
                        rotated_gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)
                    )
                    if len(faces) > 0:
                        face = max(faces, key=lambda f: f[2] * f[3])
                        tilt = self._compute_tilt(rotated_gray)
                        corrected_gray = self._apply_tilt(rotated_gray, tilt)
                        corrected_rgb = self._apply_tilt(
                            self._rotate90(img_np, angle), tilt
                        )
                        if angle != 0 or abs(tilt) > 0.5:
                            print(
                                f"[Soya: AnimeBlink] Batch {i} 회전 보정: "
                                f"{angle}° + tilt {tilt:.1f}°"
                            )
                        found = True
                        break

                if not found:
                    print(f"[Soya: AnimeBlink] Batch {i}: 얼굴 인식 실패")
                    is_closed_list.append(False)
                    openness_list.append(0.0)
                    corrected_imgs.append(img)
                    continue

            img_h, img_w = corrected_gray.shape
            (lx1, ly1, lx2, ly2), (rx1, ry1, rx2, ry2) = self._eye_rois(
                face, img_h, img_w
            )

            left_open = self._calc_openness(corrected_gray, lx1, ly1, lx2, ly2)
            right_open = self._calc_openness(corrected_gray, rx1, ry1, rx2, ry2)

            best_open = max(left_open, right_open)
            is_closed = bool(best_open < threshold)

            print(
                f"[Soya: AnimeBlink] Batch {i}: "
                f"L={left_open:.4f}, R={right_open:.4f}, "
                f"MAX={best_open:.4f} | threshold={threshold:.3f}, closed={is_closed}"
            )

            is_closed_list.append(is_closed)
            openness_list.append(best_open)
            corrected_imgs.append(
                torch.from_numpy(corrected_rgb.astype(np.float32) / 255.0)
            )

        # IMAGE 출력은 배치 텐서 [B, H, W, C]로 스택
        return (is_closed_list, openness_list, torch.stack(corrected_imgs))
