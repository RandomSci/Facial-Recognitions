from ultralytics import YOLO
import cv2
import face_recognition
from torch_snippets import *

def encode_face(img_path):
    """Encodes a face from an image file path."""
    img = face_recognition.load_image_file(img_path)
    encodings = face_recognition.face_encodings(img)
    if len(encodings) > 0:
        return encodings[0]  
    else:
        raise ValueError(f"No face detected in {img_path}")

def compare(face_encoding1, face_encoding2, tolerance=0.4):
    """Compares two face encodings with a specified tolerance."""
    return face_recognition.compare_faces([face_encoding1], face_encoding2, tolerance=tolerance)[0]

known_images = [
    r"/home/zkllmt/Documents/AI Section/Datasets/Facial_Recognition_Custom_Dataset/train/images/train/eK3MiIuz.jpeg",
    r"/home/zkllmt/Documents/AI Section/Datasets/Facial_Recognition_Custom_Dataset/train/images/train/yzT-J9oK.jpeg",
    r"/home/zkllmt/Documents/AI Section/Datasets/Facial_Recognition_Custom_Dataset/test/images/person2/Selwyn.jpg"
]
known_names = ["charmae", "charmae", "Selwyn"]

try:
    known_encodings = [encode_face(img) for img in known_images]
except ValueError as e:
    print(e)
    exit()

model = YOLO("yolo11n.pt")

cam = cv2.VideoCapture(2)
cv2.namedWindow("Predicted Frame", cv2.WINDOW_NORMAL)

try:
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        results = model.predict(frame, conf=0.5)

        for result in results[0].boxes:
            cls = int(result.cls[0])
            label = model.names[cls]
            if label == "person":
                x1, y1, x2, y2 = map(int, result.xyxy[0])  
                cropped_face = frame[y1:y2, x1:x2]

                cv2.imshow("Cropped Face", cropped_face)

                cropped_face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)

                try:
                    face_encoding = face_recognition.face_encodings(cropped_face_rgb)
                    if face_encoding:
                        face_encoding = face_encoding[0]
                        print(f"Comparing face encoding: {face_encoding}")  

                        matched = False
                        for known_encoding, name in zip(known_encodings, known_names):
                            if compare(known_encoding, face_encoding):
                                label = name
                                matched = True
                                print(f"Match found: {name}")  
                                break

                        if not matched:
                            print("No match found.")  
                except Exception as e:
                    print(f"Face encoding failed: {e}")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Predicted Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cam.release()
    cv2.destroyAllWindows()