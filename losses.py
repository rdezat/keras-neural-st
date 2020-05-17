# -*- coding: utf-8 -*-
"""
Created on Fri May 15 18:02:31 2020

@author: ricard.deza.tripiana
"""
import tensorflow as tf

import utils

# La "pèrdua d'estil" està dissenyada per mantenir l'estil de la imatge de referència a la imatge generada. 
# Es basa en la matriu de gram (que capturen l'estil) dels mapes de característiques de la imatge de referència d'estil i de la imatge generada.
def style_loss(style, combination, img_ncols, img_nrows):
    # Calculem la matriu de gram de la imatge de referència d'estil
    S = utils.gram_matrix(style)
    # Calculem la matriu de gram de la imatges de contingut
    C = utils.gram_matrix(combination)
    # Definim el canals
    channels = 3
    # Definim la mida
    size = img_nrows * img_ncols
    # Retornem la pèrdua d'estil
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

# Funció de pèrdua auxiliar dissenyada per mantenir el "contingut" de la imatge base a la imatge generada
def content_loss(base, combination):
  return tf.reduce_sum(tf.square(combination - base))

# La tercera funció de pèrdua, pèrdua de variació total, dissenyada per mantenir la imatge generada coherent localment
def total_variation_loss(x, img_ncols, img_nrows):
  a = tf.square(
      x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
  b = tf.square(
      x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
  return tf.reduce_sum(tf.pow(a + b, 1.25))

@tf.function
def compute_loss_and_grads(combination_image, base_image, style_reference_image, feature_extractor, 
                           content_weight, style_weight, total_variation_weight, img_ncols, img_nrows):
    with tf.GradientTape() as tape:
        loss, cl, sl, tl = compute_loss(combination_image, base_image, style_reference_image, feature_extractor,
                            content_weight, style_weight, total_variation_weight, img_ncols, img_nrows)
    grads = tape.gradient(loss, combination_image)
    return loss, grads, cl, sl, tl

def compute_loss(combination_image, base_image, style_reference_image, feature_extractor,
                 content_weight, style_weight, total_variation_weight, img_ncols, img_nrows):
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
    cl = content_weight * content_loss(base_image_features, combination_features)
    loss = loss + cl
    # Add style loss
    sl = 0
    for layer_name in style_layer_names:
      layer_features = features[layer_name]
      style_reference_features = layer_features[1, :, :, :]
      combination_features = layer_features[2, :, :, :]
      sl_layer = (style_weight / len(style_layer_names)) * style_loss(style_reference_features, combination_features, img_ncols, img_nrows)
      sl += sl_layer
      loss += sl_layer
    
    # Add total variation loss
    tl = total_variation_weight * total_variation_loss(combination_image, img_ncols, img_nrows)
    loss += tl
    return loss, cl, sl, tl