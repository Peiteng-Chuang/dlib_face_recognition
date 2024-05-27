import cv2
import numpy as np
import dlib

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

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('C:\\Users\\payten\\Desktop\\project_file\\dlib_face_recognition\\shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    faces = detector(frame)
    for face in faces:
        landmarks = predictor(frame, face)
        landmarks = np.array([(p.x, p.y) for p in landmarks.parts()])
    
    # 應用化妝效果（例如口紅）
        frame = apply_lipstick(frame, landmarks, (0, 0, 255))  # 紅色口紅

    
    # cv2.imshow('dlib_plus_camera', frame)
    cv2.imshow('dlib VR Makeup', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        print("Detected 'q' key press. Exiting loop.")
        break


cap.release()
cv2.destroyAllWindows()