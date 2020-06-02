import numpy as np
import cv2 as cv


rostros = cv.CascadeClassifier('./trained_classifiers/haarcascade_frontalface_default.xml')
ojos = cv.CascadeClassifier('./trained_classifiers/haarcascade_eye.xml')
upds = cv.CascadeClassifier('./trained_classifiers/cascade3.xml')

# cambiar aqui para diferente camara
cap = cv.VideoCapture(0)
# print('Abriendo Camara')
while True:
    ret, img = cap.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    caras = rostros.detectMultiScale(gray, 1.3, 5)

    logo = upds.detectMultiScale(gray, 50, 50)
    for (x, y, w, h) in logo:
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(img,'UPDS',(x-w,y-h), font, 1, (255,0,0), 3, cv.LINE_AA)
        cv.rectangle(img, (x, y), (x+w+50, y+h+50), (255, 0, 0),5)

    for (x, y, w, h) in caras:
        cv.rectangle(img, (x, y), (x+w, y + h), (0, 255, 0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = ojos.detectMultiScale(roi_gray)
        for (ox, oy, ow, oh) in eyes:
            cv.rectangle(roi_color, (ox, oy), (ox+ow, oy+oh), (0, 0, 255), 2)

    cv.imshow('foto1', img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()
