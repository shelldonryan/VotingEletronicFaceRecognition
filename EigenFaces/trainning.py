import cv2
import os
import numpy as np

eigenFaces = cv2.face.EigenFaceRecognizer.create()

def getImageComId():
    paths = [os.path.join('../photos/photosForTrainning', f) for f in os.listdir('../photos/photosForTrainning')]

    faces = []
    ids = []

    for pathImages in paths:
        frameFace = cv2.cvtColor(cv2.imread(pathImages), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(pathImages)[-1].split('.')[1])
        print(id)
        ids.append(id)
        faces.append(frameFace)

    return np.array(ids), faces

ids, faces = getImageComId()

print("Treinando...")

eigenFaces.train(faces, ids)
eigenFaces.write("classificator.yml")

print("Treinando Concluído")
