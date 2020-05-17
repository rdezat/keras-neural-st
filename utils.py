# -*- coding: utf-8 -*-
"""
Created on Wed May 13 19:55:52 2020

@author: ricard.deza.tripiana
"""

import os
import sys
import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.applications import vgg19

# Funció per obrir, redimensionr i formatejar imatges en tensors adequats
def preprocess_image(image_path, img_ncols, img_nrows):
    # Obrim la imatges i redimensionem
    img = keras.preprocessing.image.load_img(image_path, target_size=(img_nrows, img_ncols))
    # Convertim la imatges en un array
    img = keras.preprocessing.image.img_to_array(img)
    # Inserim una dimensió al array de la imatge
    img = np.expand_dims(img, axis=0)
    # Les imatges es converteixen de RGB a BGR i, a continuació, cada canal de color 
    # està centrat en zero respecte al conjunt de dades ImageNet, sense fer-ne escala.
    img = vgg19.preprocess_input(img)
    # Retornem la imatges convertida en Tensor
    return tf.convert_to_tensor(img)

# Funció per convertir un tensor en una imatges vàlida
def deprocess_image(x, img_ncols, img_nrows):
    # Donem forma a la matriu de la imatges sense canviar les dades
    x = x.reshape((img_nrows, img_ncols, 3))
    # Eliminem el centre zero amb el píxel mitjà
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    # Per valors majors a 255 i menors a 0, s'aplica el màxim i el mínim respectivament.
    # Convertim a int
    x = np.clip(x, 0, 255).astype('uint8')
    # Retornem el array de la imatge vàlida
    return x

# Calculem la matriu de gram d'un tensor d'imatges (producte exterior característic)
def gram_matrix(x):
    # Transposem el tensor
    x = tf.transpose(x, (2, 0, 1))
    # Donem forma a la matriu de la imatges sense canviar les dades
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    # Calculem el producte exterior entre les característiques i la seva transposta
    gram = tf.matmul(features, tf.transpose(features))
    # Retornem la matriu de gram
    return gram

def check_paths(output_path):
    try:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    except OSError as e:
        print(e)
        sys.exit(1)