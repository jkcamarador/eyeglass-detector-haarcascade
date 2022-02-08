import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeglass_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
font = cv2.FONT_HERSHEY_COMPLEX

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, black_and_white) = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    faces_bw = face_cascade.detectMultiScale(black_and_white, 1.1, 4)

    if (len(faces) == 0 and len(faces_bw) == 0):
        cv2.putText(img, "No face detected.", (10,30), font, 0.7, (255, 0, 0), 1, cv2.LINE_AA)
    else:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            eyes = eyeglass_cascade.detectMultiScale(roi_gray)

        if (len(eyes) == 0):
            cv2.putText(img, "Eyeglass detected.", (x, y - 10), font, 0.7, (255,255,224), 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "No Eyeglass detected.", (x, y - 10), font, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.imshow('Eyeglass Detection', img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
