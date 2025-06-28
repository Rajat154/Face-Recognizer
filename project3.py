import cv2
import os

# Load Haar cascade with absolute path
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

datasets = 'datasets'
sub_data = 'Rajat'
path = os.path.join(datasets, sub_data)
if not os.path.isdir(path):
    os.makedirs(path)

(width, height) = (130, 100)
webcam = cv2.VideoCapture(0)  # Use 0 instead of 1

count = 1
while count < 101:
    print(f"Capturing image: {count}")
    ret, im = webcam.read()
    if not ret:
        print("Failed to capture image.")
        break

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y+h, x:x+w]
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite(f'{path}/{count}.png', face_resize)
        count += 1

    cv2.imshow('OpenCV', im)
    if cv2.waitKey(10) == 27:  # ESC key
        break

webcam.release()
cv2.destroyAllWindows()
