import torch
import numpy as np
import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# face_landmarker 태스크 파일 다운로드 경로 (custom_nodes/comfyui-soya-custom 폴더 내부로 지정)
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "face_landmarker.task")
model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# 노드 로드 시점에 모델이 없으면 다운로드 (기존 legacy solutions 대신 tasks API 사용)
if not os.path.exists(model_path):
    print(f"Downloading MediaPipe FaceLandmarker model to {model_path}...")
    urllib.request.urlretrieve(model_url, model_path)

class MediaPipeEyeDetector:
    def __init__(self):
        # MediaPipe Tasks API 모드로 초기화 (Python 3.12+ 등에서 solutions 어트리뷰트 에러 방지)
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)}}
    
    RETURN_TYPES = ("BOOLEAN", "FLOAT")
    RETURN_NAMES = ("is_closed", "ear_value")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "detect_eye_state"
    CATEGORY = "Custom/Face"

    def detect_eye_state(self, image):
        is_closed_list = []
        ear_value_list = []
        
        # ComfyUI의 image는 [B, H, W, C] 형태의 텐서이므로 B(배치)만큼 반복
        for i, img in enumerate(image):
            img_np = (img.cpu().numpy() * 255).astype(np.uint8)
            height, width, _ = img_np.shape
            
            # mediapipe 최신 Tasks API용 이미지 객체 생성
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
            
            # 랜드마크 추출
            detection_result = self.detector.detect(mp_image)
            
            if not detection_result.face_landmarks:
                print(f"[Soya: EyeDetector] ⚠️ Batch {i} - 얼굴을 찾지 못했습니다! (Crop이 너무 타이트하거나 얼굴이 아님). ear=0.0 반환")
                is_closed_list.append(False)
                ear_value_list.append(0.0)
                continue
                
            landmarks = detection_result.face_landmarks[0]
            
            # 왼쪽 눈(이미지 상 우측에 위치) 좌/우 끝점 좌표 및 상/하 끝점 좌표 
            # (해당 인덱스들은 MediaPipe Face Mesh의 Right Eye(진짜 눈은 왼쪽) 구조를 참고한 것입니다)
            top = landmarks[386]
            bottom = landmarks[374]
            left = landmarks[362]
            right = landmarks[263]
            
            # MediaPipe의 랜드마크 x, y는 0~1 사이로 정규화(normalized)된 값입니다.
            # 정확한 거리 및 비율 도출을 위해 이미지의 실제 폭과 높이를 곱해 픽셀 단위로 계산합니다.
            def calc_dist(pt1, pt2):
                return np.sqrt(((pt1.x - pt2.x) * width) ** 2 + ((pt1.y - pt2.y) * height) ** 2)

            v_dist = calc_dist(top, bottom)
            h_dist = calc_dist(left, right)
            
            ear = v_dist / (h_dist + 1e-6)
            
            # EAR 값이 0.15~0.2 이하로 떨어지면 눈을 감은 것으로 판단
            is_closed = True if ear < 0.18 else False

            # 검증을 위해 터미널에 명확한 prefix와 함께 로그 출력
            print(f"\n[Soya: EyeDetector] --- Batch {i} Result ({width}x{height}) ---")
            print(f"[Soya: EyeDetector] Top/Bottom : ({top.x:.3f}, {top.y:.3f}) / ({bottom.x:.3f}, {bottom.y:.3f}) -> v_dist: {v_dist:.2f} px")
            print(f"[Soya: EyeDetector] Left/Right : ({left.x:.3f}, {left.y:.3f}) / ({right.x:.3f}, {right.y:.3f}) -> h_dist: {h_dist:.2f} px")
            print(f"[Soya: EyeDetector] EAR: {ear:.4f} | is_closed: {is_closed}")
            
            is_closed_list.append(is_closed)
            ear_value_list.append(float(ear))
            
        return (is_closed_list, ear_value_list)

NODE_CLASS_MAPPINGS = {"MediaPipeEyeDetector": MediaPipeEyeDetector}

