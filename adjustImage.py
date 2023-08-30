import cv2
import os
from imutils import paths
import shutil

pasta = 'photos/ShelldonRyan'
tituloEleitoral = "049938681210"

def listNegImagem():
    imagemPath = list(paths.list_images(pasta))
    numero = 1
    if not os.path.exists('photos/photosForTrainning'):
        os.makedirs('photos/photosForTrainning')

    for i in imagemPath:
        rename = i.replace(i, "photos/photosForTrainning/" + "people." + str(tituloEleitoral) + "." + str(numero) + ".jpg")
        fotoName = rename.split("/")[2]
        shutil.copy(i, rename)
        img = cv2.imread("photos/photosForTrainning/" + fotoName, cv2.IMREAD_GRAYSCALE)
        resized_image = cv2.resize(img, (220, 220))
        cv2.imwrite("photos/photosForTrainning/" + fotoName, resized_image)

        print(fotoName)

        numero += 1

listNegImagem()