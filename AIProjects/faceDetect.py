import cv2 as cv

cam = cv.VideoCapture(0)

cv.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break

    k = cv.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    image_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face_cascade = cv.CascadeClassifier('./cascades/haarcascade_frontalface_alt.xml')
    detected_faces = face_cascade.detectMultiScale(image_gray)

    for (column, row, width, height) in detected_faces:
        cv.rectangle(
            frame,
            (column, row),
            (column + width, row + height),
            (0, 0, 255),
            2
        )

    cv.imshow('test', frame)

cam.release()

cv.destroyAllWindows()


