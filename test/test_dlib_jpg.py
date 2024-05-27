import cv2
import numpy as np
import dlib

# 定義化妝效果函數
def apply_eyeshadow(image, landmarks, color):
    left_eye_points = landmarks[36:42]
    right_eye_points = landmarks[42:48]
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [left_eye_points, right_eye_points], color)
    result = cv2.addWeighted(image, 1, mask, 0.4, 0)
    return result

def apply_blush(image, landmarks, color):
    left_cheek = landmarks[1:3]
    right_cheek = landmarks[14:16]
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [left_cheek, right_cheek], color)
    result = cv2.addWeighted(image, 1, mask, 0.4, 0)
    return result

def apply_lipstick(image, landmarks, color):
    lips_points = landmarks[48:61]  # 嘴唇的點
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [lips_points], color)
    result = cv2.addWeighted(image, 1, mask, 0.4, 0)
    return result


# 載入dlib的面部偵測器和形狀預測模型
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


# 讀取圖像
image = cv2.imread('face.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 偵測面部
faces = detector(gray)
for face in faces:
    landmarks = predictor(gray, face)
    landmarks = np.array([(p.x, p.y) for p in landmarks.parts()])
    
    # 應用化妝效果（例如口紅）
    image = apply_lipstick(image, landmarks, (50, 50, 100))  # 紅色口紅

# 顯示結果
cv2.imshow('dlib VR Makeup', image)
cv2.waitKey(0)
cv2.destroyAllWindows()