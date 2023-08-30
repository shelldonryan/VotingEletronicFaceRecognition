import cv2

cap = cv2.VideoCapture(0)

detectorFace = cv2.CascadeClassifier("../haarcascade_frontalface_default.xml")
recognitionFace = cv2.face.LBPHFaceRecognizer.create()
recognitionFace.read("classificator.yml")
largura, altura = 220, 220


while True:
    ret, frame = cap.read()

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    matches = detectorFace.detectMultiScale(gray_image, scaleFactor=1.3, minSize=(30, 30))

    for (x, y, l, a) in matches:
        frameFace = cv2.resize(gray_image[y:y + a, x:x + l], (largura, altura))
        cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 255), 2)

        ids, authenticator = recognitionFace.predict(frameFace)
        print(ids)

        if ids == 49938681210:
            name = "APTO A VOTAR"
        elif ids != 49938681210:
            name = "DESCONHECIDO"

        cv2.putText(frame, name, (x, y+(a+20)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255))

    cv2.imshow('Captura de Imagem', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
