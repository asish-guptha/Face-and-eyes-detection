import cv2

# Load the required trained XML classifiers for face and eyes detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Capture frames from a camera
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Loop runs if capturing has been initialized.
while True:

    # Read frames from a camera
    ret, img = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces of different sizes in the input image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # To draw a rectangle around the face
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            # To draw a rectangle around the eyes
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 127, 255), 2)

    # Display the image in a window
    cv2.imshow('Face and Eye Detection', img)

    # Wait for the 'Esc' key to stop the loop
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
