import os
from glob import glob
import cv2
import numpy as np
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POSITIVE_PATH = os.path.join(BASE_DIR, "dataset/positives")
NEGATIVE_PATH = os.path.join(BASE_DIR, "dataset/negatives")
ANNOTATION_PATH = "annotations"
VEC_DIR = "vec"
CASCADE_DIR = "cascade"
POSITIVES_FILE = os.path.join(ANNOTATION_PATH, "positives.txt")
VEC_FILE = os.path.join(VEC_DIR, "positives.vec")


def list_images(directory: str, extensions: tuple = (".jpg", ".jpeg", ".png")) -> list:
    files = []
    for ext in extensions:
        files.extend(glob(os.path.join(directory, f"*.{ext}")))
    return sorted(files)


def preparar_imagens():
    #imagens = list_images()

    # Lista para armazenar os caminhos das imagens
    positivas = []
    negativas = []

    # Adicione aqui os caminhos das suas imagens positivas
    pasta_positivas = "C:/Users/luana/OneDrive/Área de Trabalho/piriquitos"
    for i in range(30):  # para 30 imagens positivas
        caminho = pasta_positivas + f"passaro{i + 1}.jpg"
        positivas.append(caminho)

    # Adicione aqui os caminhos das suas imagens negativas
    pasta_negativas = "C:/Users/luana/OneDrive/Área de Trabalho/sem_piriquitos"
    for i in range(50):  # para 50 imagens negativas
        caminho = pasta_negativas + f"negativa{i + 1}.jpg"
        negativas.append(caminho)

    # Criar arquivo de descrição das imagens positivas
    with open('positives.txt', 'w') as f:
        for img_path in positivas:
            f.write(f'{img_path} 1 0 0 50 50\n')

    # Criar arquivo de descrição das imagens negativas
    with open('negatives.txt', 'w') as f:
        for img_path in negativas:
            f.write(f'{img_path}\n')


def treinar_detector():
    # Configuração do classificador cascade
    params = {
        'cascade_name': 'cascade_bird.xml',
        'width': 24,
        'height': 24,
        'stage_num': 10,
        'min_hit_rate': 0.995,
        'max_false_alarm_rate': 0.5,
        'weight_trim_rate': 0.95,
        'max_depth': 1,
        'max_weak_count': 100
    }

    # Criar e treinar o classificador
    cascade = cv2.CascadeClassifier()
    cascade.train(
        'positives.vec',
        'negatives.txt',
        params['cascade_name'],
        params['width'],
        params['height'],
        params['stage_num']
    )


def detectar_passaros(imagem_path):
    detector = cv2.CascadeClassifier('cascade_bird.xml')
    imagem = cv2.imread(imagem_path)
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    passaros = detector.detectMultiScale(
        cinza,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(24, 24)
    )

    for (x, y, w, h) in passaros:
        cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Passaros Detectados', imagem)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Executar o código
#preparar_imagens()  # Primeiro, prepare as imagens
#treinar_detector()  # Depois, treine o detector
#detectar_passaros('teste.jpg')  # Por fim, teste

if __name__ == '__main__':
    print(POSITIVE_PATH)
    #images = list_images(POSITIVE_PATH)
    positivas = list_images(POSITIVE_PATH)
    pasta_positivas = POSITIVE_PATH
    for i in range(30):  # para 30 imagens positivas
        caminho = pasta_positivas + f"passaro{i + 1}.jpg"
        positivas.append(caminho)

    #for image_path in images:
      #  print(image_path)
      #  filename = os.path.basename(image_path)
