import cv2

cap = cv2.VideoCapture(0)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow('carema test', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        print("Detected 'q' key press. Exiting loop.")
        break

cap.release()
cv2.destroyAllWindows()
