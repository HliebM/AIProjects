import cv2 as cv
import face_recognition
import webbrowser
import os


cam = cv.VideoCapture(0)

cv.namedWindow("Video")

face_cascade = cv.CascadeClassifier('./cascades/haarcascade_frontalface_alt.xml')

Hlieb_image = face_recognition.load_image_file("img/Hlieb.jpg")
Elon_image = face_recognition.load_image_file("img/Elon.jpg")
Thomas_image = face_recognition.load_image_file("img/Thomas.png")
Romal_image = face_recognition.load_image_file("img/Romal.png")


known_face_encodings = [
    face_recognition.face_encodings(Hlieb_image)[0],
    face_recognition.face_encodings(Elon_image)[0],
    face_recognition.face_encodings(Thomas_image)[0],
    face_recognition.face_encodings(Romal_image)[0],
]
known_face_names = ['Hlieb', 'Elon', 'Thomas', 'Romal']

colors = (0, 0, 255)
webBrowserNotOpened = False


while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break

    k = cv.waitKey(1)

    cv.imshow('Video', frame)

    image_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(image_gray)



    for (column, row, width, height) in detected_faces:
        try:
            unknownNotSavedImage = frame[row:row + height, column:column + width]
            cv.imwrite("./img/cropped_image.jpg", unknownNotSavedImage)

            unknownSavedImage = cv.imread("./img/cropped_image.jpg")
            os.remove("./img/cropped_image.jpg")

            unknown_encoding = face_recognition.face_encodings(unknownSavedImage)[0]

            a = 0
            nameOfSeenPerson = "Someone"
            while a <= len(known_face_encodings):
                a += 1
                results = face_recognition.compare_faces([known_face_encodings[len(known_face_encodings) - a]], unknown_encoding)
                if results[0]:
                    nameOfSeenPerson = known_face_names[len(known_face_encodings) - a]
                    a = 0
                    break
                else:
                    pass

            # If you want to show cropped Video--->
            # cv.imshow('Face', unknownNotSavedImage)
        except:
            results = [False]

        if results[0]:
            colors = (0, 255, 0)
            if webBrowserNotOpened:
                webbrowser.open('http://127.0.0.1:8000/signIn')
                webBrowserNotOpened = False
                if k % 256 == 27:
                    # ESC pressed
                    print("Escape hit, closing...")
                    break
            else:
                if k % 256 == 27:
                    # ESC pressed
                    print("Escape hit, closing...")
                    break
        else:
            colors = (0, 0, 255)
            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break

        cv.rectangle(
            frame,
            (column, row),
            (column + width, row + height),
            colors,
            2
        )
        cv.putText(
            frame,
            f"{nameOfSeenPerson}",
            (column + width, row + height + 20),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            colors,
            2
        )
        cv.imshow('Video', frame)
        nameOfSeenPerson = "Someone"
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

    try:
        print(f"There are {detected_faces.size / 4} people")
    except:
        print("There are 0 people")


cam.release()

cv.destroyAllWindows()

