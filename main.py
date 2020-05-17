# -*- coding: utf-8 -*-
"""
Created on Wed May 13 19:08:11 2020

@author: ricard.deza.tripiana
"""
import argparse
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
utils.check_paths(opt.output_path)

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
base_image = utils.preprocess_image(opt.base_image_path, img_ncols, opt.img_nrows)
style_reference_image = utils.preprocess_image(opt.style_reference_image_path, img_ncols, opt.img_nrows)
combination_image = tf.Variable(utils.preprocess_image(opt.base_image_path, img_ncols, opt.img_nrows))

# Entrenament
# Per cada època
for i in range(opt.n_epochs):

    loss, grads, cl, sl, tl = losses.compute_loss_and_grads(combination_image, base_image, style_reference_image, feature_extractor,
                                                opt.content_weight, opt.style_weight, opt.total_variation_weight,
                                                img_ncols, opt.img_nrows)
    
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
