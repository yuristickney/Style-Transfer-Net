# coding: utf-8
import keras
from keras.preprocessing.image import load_img, img_to_array

import scipy.optimize as optimize
import scipy.misc as sm
import time

import numpy as np
from keras.applications import vgg19

from matplotlib import pyplot as plt

from keras import backend as K


target_image_path = r'Y:\Documents\StyleNetwork\City_1.jpg'
style_reference_image_path = r'Y:\Documents\StyleNetwork\Graphic_1.jpg'
start_image_path = r'Y:\Documents\StyleNetwork\Space_1.jpg'

width, height = load_img(target_image_path).size

img_height  = 400
img_width = int(width * img_height / height)


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


def deprocess_image(img):
    # Remove zero-center by mean pixel
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype('uint8')
    return img


def content_loss(base, combination):
    return 0.5 * K.sum(K.square(combination - base))


def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


def total_variation_loss(x):
    a = K.square(x[:, :img_height - 1, :img_width - 1, :] - x[:, 1:, :img_width - 1, :])
    b = K.square(x[:, :img_height - 1, :img_width - 1, :] - x[:, :img_height - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


def get_layers(content_matrix, style_matrix, generated_matrix):
    input_tensor = K.concatenate([content_matrix, style_matrix, generated_matrix], axis=0)
    # We build the VGG19 network with our batch of 3 images as input.
    # The model will be loaded with pre-trained ImageNet weights.
    model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)
    print('vgg Model loaded.')
    layers = {layer.name: layer.output for layer in model.layers}
    content_layer = layers['block3_conv2']
    style_layers = ['block1_conv1', 'block1_conv2',
                    'block2_conv1', 'block2_conv2',
                    'block3_conv1', 'block3_conv2',
                    'block3_conv3', 'block3_conv4',
                    'block4_conv1', 'block4_conv2',
                    'block4_conv3', 'block4_conv4',
                    'block5_conv1', 'block5_conv2',
                    'block5_conv3', 'block5_conv4']
    style_layers = [layers[layer] for layer in style_layers]

    return content_layer, style_layers


def total_loss(c_layer, s_layer, g_img):
    # Weights in the weighted average of the loss components
    total_variation_weight = 1e-4
    style_weight = 1.
    content_weight = 0.1

    content_features = c_layer[0, :, :, :]
    generated_img_features = c_layer[2, :, :, :]

    c_loss = content_loss(content_features, generated_img_features)
    s_loss = K.variable(0.)
    for layer in s_layer:
        style_features = layer[1, :, :, :]
        generated_img_features = layer[2, :, :, :]
        s_loss += style_loss(style_features, generated_img_features) * (style_weight / len(s_layer))
    v_loss = total_variation_loss(g_img)

    return content_weight * c_loss + s_loss + total_variation_weight * v_loss


def eval_loss_and_grads(x):
    x = x.reshape((1, height, width, 3))
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1].flatten().astype('float64')
    return loss_value, grad_values


class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

    
def run_module():
    # gen_img = np.random.uniform(0, 255, (1, img_height, img_width, 3)) - 128. # Noise Image
    gen_img = preprocess_image(start_image_path)  # Chosen Image

    target_image = K.constant(preprocess_image(target_image_path))
    style_reference_image = K.constant(preprocess_image(style_reference_image_path))

    # This placeholder will contain our generated image
    gen_img_placeholder = K.placeholder((1, img_height, img_width, 3))

    # We combine the 3 images into a single batch
    input_tensor = K.concatenate([target_image, style_reference_image, gen_img_placeholder], axis=0)
    loss = K.variable(0.)

    # Grab the layers needed to prepare the loss metric
    content_layer, style_layers = get_layers(target_image, style_reference_image, gen_img)

    # Define loss and gradient
    loss = total_loss(content_layer, style_layers, gen_img)
    grads = K.gradients(loss, gen_img)

    outputs = [loss]
    outputs += grads
    fetch_loss_and_grads = K.function([gen_img], outputs)

    evaluator = Evaluator()
    epochs = 40
    result_prefix = 'style_transfer_result'


    name = 'Initial_image.png'
    sm.imsave(name, gen_img)
    print('Initial image saved as', name)

    for i in range(epochs):
        print('Start of iteration', i)
        start_time = time.time()
        gen_img, min_val, info = optimize.fmin_l_bfgs_b(evaluator.loss, gen_img.flatten(), fprime=evaluator.grads, maxfun=20)
        print('Current loss value:', min_val)
        # Save current generated image
        gen_img = gen_img.copy().reshape((img_height, img_width, 3))
        gen_img = deprocess_image(gen_img)
        name = result_prefix + '_at_iteration_%d.png' % i + 1
        sm.imsave(name, gen_img)
        end_time = time.time()
        print('Image saved as', name)
        print('Iteration %d completed in %ds' % (i, end_time - start_time))

    # Content image
    plt.imshow(load_img(target_image_path, target_size=(img_height, img_width)))
    plt.figure()

    # Style image
    plt.imshow(load_img(style_reference_image_path, target_size=(img_height, img_width)))
    plt.figure()

    # Generate image
    plt.imshow(gen_img)
    plt.show()

if __name__ == '__main__':
    run_module()



