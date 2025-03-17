import threading
import cv2 
from deepface import DeepFace
import os

# Check if reference image exists
if not os.path.exists("reference.jpg"):
    print("Error: reference.jpg not found")
    exit()

# Initialize camera with error handling
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Using CAP_DSHOW for Windows

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_match = False

# Load reference image with error handling
reference_img = cv2.imread("reference.jpg")
if reference_img is None:
    print("Error: Could not load reference.jpg")
    cap.release()
    exit()

def check_face(frame):
    global face_match
    try:
        if DeepFace.verify(frame, reference_img.copy())['verified']:
            face_match = True
        else:
            face_match = False
    except Exception as e:  # Catch all exceptions
        face_match = False
        print(f"Error in face verification: {str(e)}")

while True:

    ret, frame = cap.read()

    if ret:
        if counter % 38 == 8:

            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start() 
            except ValueError:
                pass
        counter += 1

        if face_match:
            cv2.putText(frame, "MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("video", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
cv2.destroyAllWindows()
cap.release()
