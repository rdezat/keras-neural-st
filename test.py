# -*- coding: utf-8 -*-
"""
Created on Sat May 16 18:23:04 2020

@author: ricard.deza.tripiana
"""

import sys
import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg19

import utils
import losses

parser = argparse.ArgumentParser()
parser.add_argument("--base-image-path", type=str, default="images/fotografia.jpg", help="Ruta a la imatge amb el contingut a on transferir l'estil")
parser.add_argument("--style-reference-image-path", type=str, default="images/vangogh.jpg", help="Ruta a la imatge amb l'estil a transferir")
parser.add_argument("--output-path", type=str, default="images/output/", help="Ruta de les imatges de sortida")
parser.add_argument("--n-epochs", type=int, default=5000, help="Nombre d'èpoques d'entrenament")
parser.add_argument("--total-variation-weight", type=float, default=1e-6, help="Pes de la pèrdua total")
parser.add_argument("--style-weight", type=float, default=2e-6, help="Pes de la pèrdua d'estil")
parser.add_argument("--content-weight", type=float, default=2e-8, help="Pes de la pèrdua de contingut")
parser.add_argument("--img-nrows", type=int, default=400, help="Nombre de files de la imatges generada")
parser.add_argument("--initial-learning-rate", type=float, default=100., help="Taxa d'aprenentatge inicial")
parser.add_argument("--decay-steps", type=int, default=100, help="Èpoques de decaïment de la taxa d'aprenentatge")
parser.add_argument("--decay-rate", type=float, default=0.96, help="Decaïment de la taxa d'aprenentatge")
parser.add_argument("--epoch-save", type=int, default=100, help="Nombre de èpoques per desar les sortides")
opt = parser.parse_args()

# Comprovem que la ruta on es desen les sortides existeixi. En cas negatiu, es crea
try:
    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path)
except OSError as e:
    print(e)
    sys.exit(1)

# Obtenim les dimensions de la imatge amb el contingut a transferir
width, height = keras.preprocessing.image.load_img(opt.base_image_path).size
# Calculem les dimensions de les imatges generades
img_ncols = int(width * opt.img_nrows / height)

# Contruïm un model VGG19 preentrenat amb ImageNet
model = vgg19.VGG19(weights='imagenet', include_top=False)

# Obtenim les sortides simbòliques de cada capa "clau" (tenen noms únics).
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

# Configurem un model que retorni els valors d'activació de cada capa de VGG19 (com a un diccionari).
feature_extractor = keras.Model(inputs=model.inputs,
                                outputs=outputs_dict)

# Configurem l'optimitzador SGD amb la programació pel decaïment de la taxa d'aprenentatge
optimizer = keras.optimizers.SGD(
    keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=opt.initial_learning_rate,
                                                decay_steps=opt.decay_steps,
                                                decay_rate=opt.decay_rate)
)

# Preprocessem les imatges d'entrada.
# Obrim la imatges i redimensionem
base_image = keras.preprocessing.image.load_img(opt.base_image_path, target_size=(opt.img_nrows, img_ncols))
# Convertim la imatges en un array
base_image = keras.preprocessing.image.img_to_array(base_image)
# Inserim una dimensió al array de la imatge
base_image = np.expand_dims(base_image, axis=0)
# Les imatges es converteixen de RGB a BGR i, a continuació, cada canal de color 
# està centrat en zero respecte al conjunt de dades ImageNet, sense fer-ne escala.
base_image = vgg19.preprocess_input(base_image)
# Retornem la imatges convertida en Tensor
base_image = tf.convert_to_tensor(base_image)

# Obrim la imatges i redimensionem
style_reference_image = keras.preprocessing.image.load_img(opt.style_reference_image_path, target_size=(opt.img_nrows, img_ncols))
# Convertim la imatges en un array
style_reference_image = keras.preprocessing.image.img_to_array(style_reference_image)
# Inserim una dimensió al array de la imatge
style_reference_image = np.expand_dims(style_reference_image, axis=0)
# Les imatges es converteixen de RGB a BGR i, a continuació, cada canal de color 
# està centrat en zero respecte al conjunt de dades ImageNet, sense fer-ne escala.
style_reference_image = vgg19.preprocess_input(style_reference_image)
# Retornem la imatges convertida en Tensor
style_reference_image = tf.convert_to_tensor(style_reference_image)

# Obrim la imatges i redimensionem
combination_image = keras.preprocessing.image.load_img(opt.base_image_path, target_size=(opt.img_nrows, img_ncols))
# Convertim la imatges en un array
combination_image = keras.preprocessing.image.img_to_array(combination_image)
# Inserim una dimensió al array de la imatge
combination_image = np.expand_dims(combination_image, axis=0)
# Les imatges es converteixen de RGB a BGR i, a continuació, cada canal de color 
# està centrat en zero respecte al conjunt de dades ImageNet, sense fer-ne escala.
combination_image = vgg19.preprocess_input(combination_image)
# Retornem la imatges convertida en Tensor
combination_image = tf.Variable(tf.convert_to_tensor(combination_image))

# Entrenament
# Per cada època
for i in range(opt.n_epochs):
    with tf.GradientTape() as tape:
        input_tensor = tf.concat([base_image, style_reference_image, combination_image], axis=0)
        features = feature_extractor(input_tensor)
    
        # Initialize the loss
        loss = tf.zeros(shape=())
        
        # La capa utiltzada per la pèrdua de contingut
        content_layer_name = 'block5_conv2'
        # Llista de les capes utilitzades per la pèrdua d'estil
        style_layer_names = [
          'block1_conv1', 'block2_conv1',
          'block3_conv1', 'block4_conv1',
          'block5_conv1'
        ]
        
        # Add content loss
        layer_features = features[content_layer_name]
        base_image_features = layer_features[0, :, :, :]
        combination_features = layer_features[2, :, :, :]
        cl = opt.content_weight * tf.reduce_sum(tf.square(combination_features - base_image_features))
        loss = loss + cl
        
        
        # Add style loss
        sl = 0
        for layer_name in style_layer_names:
            layer_features = features[layer_name]
            style_reference_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            # Calculem la matriu de gram de la imatge de referència d'estil
            S = utils.gram_matrix(style_reference_features)
            # Calculem la matriu de gram de la imatges de contingut
            C = utils.gram_matrix(combination_features)
            # Definim el canals
            channels = 3
            # Definim la mida
            size = opt.img_nrows * img_ncols
            # Retornem la pèrdua d'estil
            sl_layer = (opt.style_weight / len(style_layer_names)) * tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))
            sl += sl_layer
            loss += sl_layer
        
        # Add total variation loss
        a = tf.square(combination_image[:, :opt.img_nrows - 1, :img_ncols - 1, :] - combination_image[:, 1:, :img_ncols - 1, :])
        b = tf.square(combination_image[:, :opt.img_nrows - 1, :img_ncols - 1, :] - combination_image[:, :opt.img_nrows - 1, 1:, :])
        tl = opt.total_variation_weight * tf.reduce_sum(tf.pow(a + b, 1.25))
        loss += tl
        
    grads = tape.gradient(loss, combination_image)
    
    # Apliquem els gradients al optimitzador
    optimizer.apply_gradients([(grads, combination_image)])
    # Cada epoch_save èpoques
    if i % opt.epoch_save == 0:
        # Es grava log per consola amb el nombre de època i la pèrdua total
        print('Iteration %d: loss=%.2f, content_loss:%.2f, style_loss:%.2f, total_variation_loss:%.2f' % (i, loss, cl, sl, tl))
        # Es "deprocessa" la imatges generada
        img = utils.deprocess_image(combination_image.numpy(), img_ncols, opt.img_nrows)
        # Es nombra la imatges generada
        fname = opt.output_path + 'output_generated_at_iteration_%d.png' % i
        # Es desa la imatges generada en la ruta inidcada per paràmetre
        keras.preprocessing.image.save_img(fname, img)