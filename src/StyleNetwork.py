# coding: utf-8
from keras.preprocessing.image import load_img, img_to_array

import matplotlib as plt

import numpy as np
from keras.applications import vgg19

import scipy.optimize as so
from scipy.misc import imsave
import time

from keras import backend as K


class StyleNetwork:
    def __init__(self, content_img_path, style_img_path, starting_img_path=None):
        self.content_img_path = content_img_path
        self.style_img_path = style_img_path
        width, height = load_img(content_img_path).size

        self.img_height = 400
        self.img_width = int(width * self.img_height / height)

        if starting_img_path is not None:
            self.generated_img = self.preprocess_image(starting_img_path)
        else:
            self.generated_img = np.random.uniform(0, 255, (1, self.img_height, self.img_width, 3)) - 128.
        self.generated_img = self.generated_img.flatten()
        self.combination_image = K.placeholder((1, self.img_height, self.img_width, 3))
        self.model_layers = self._create_tensor()

    def preprocess_image(self, image_path):
        img = load_img(image_path, target_size=(self.img_height, self.img_width))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = vgg19.preprocess_input(img)
        return img

    def deprocess_image(self, x):
        # Remove zero-center by mean pixel
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        # 'BGR'->'RGB'
        x = x[:, :, ::-1]
        x = np.clip(x, 0, 255).astype('uint8')
        return x

    def content_loss(self, base, combination):
        return K.sum(K.square(combination - base))

    def gram_matrix(self, x):
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
        gram = K.dot(features, K.transpose(features))
        return gram

    def style_loss(self, style, combination):
        S = self.gram_matrix(style)
        C = self.gram_matrix(combination)
        channels = 3
        size = self.img_height * self.img_width
        return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

    def total_variation_loss(self, x):
        a = K.square(x[:, :self.img_height - 1, :self.img_width - 1, :] - x[:, 1:, :self.img_width - 1, :])
        b = K.square(x[:, :self.img_height - 1, :self.img_width - 1, :] - x[:, :self.img_height - 1, 1:, :])
        return K.sum(K.pow(a + b, 1.25))

    class Evaluator(object):
        def __init__(self, fetch_loss_and_grads, img_height, img_width):
            self.loss_value = None
            self.grads_values = None
            self.fetch_loss_and_grads = fetch_loss_and_grads
            self.img_height = img_height
            self.img_width = img_width

        def loss(self, x):
            assert self.loss_value is None
            x = x.reshape((1, self.img_height, self.img_width, 3))
            outs = self.fetch_loss_and_grads([x])
            loss_value = outs[0]
            grad_values = outs[1].flatten().astype('float64')
            self.loss_value = loss_value
            self.grad_values = grad_values
            return self.loss_value

        def grads(self, x):
            assert self.loss_value is not None
            grad_values = np.copy(self.grad_values)
            self.loss_value = None
            self.grad_values = None
            return grad_values

    def _create_tensor(self):
        target_image = K.constant(self.preprocess_image(self.content_img_path))
        style_image = K.constant(self.preprocess_image(self.style_img_path))
        input_tensor = K.concatenate([target_image, style_image, self.combination_image], axis=0)
        # We build the VGG19 network with our batch of 3 images as input.
        # The model will be loaded with pre-trained ImageNet weights.
        model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)
        print('vgg Model loaded.')
        return {layer.name: layer.output for layer in model.layers}

    def generate_image(self, style_weight=1., content_weight=0.05, total_variation_weight=1e-4):
        # Name of layer used for content loss
        content_layer = 'block2_conv2'
        # Name of layers used for style loss
        style_layers = ['block1_conv2', 'block2_conv2',
                        'block3_conv4', 'block4_conv4',
                        'block5_conv4']

        # Define the loss by adding all components to a `loss` variable
        loss = K.variable(0.)
        layer_features = self.model_layers[content_layer]
        target_image_features = layer_features[0, :, :, :]
        combination_features = layer_features[2, :, :, :]
        loss += content_weight * self.content_loss(target_image_features, combination_features)

        for layer_name in style_layers:
            layer_features = self.model_layers[layer_name]
            style_reference_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            sl = self.style_loss(style_reference_features, combination_features)
            loss += (style_weight / len(style_layers)) * sl

        loss += total_variation_weight * self.total_variation_loss(self.combination_image)
        grads = K.gradients(loss, self.combination_image)[0]
        fetch_loss_and_grads = K.function([self.combination_image], [loss, grads])

        evaluator = self.Evaluator(fetch_loss_and_grads, self.img_height, self.img_width)

        result_prefix = r'.\Results\style_transfer_result'
        iterations = 100

        img = self.generated_img.copy().reshape((self.img_height, self.img_width, 3))
        img = self.deprocess_image(img)
        fname = result_prefix + '_at_iteration_initial.png'
        imsave(fname, img)
        print('Image saved as', fname)

        # Run scipy-based optimization (L-BFGS) over the pixels of the generated image
        # so as to minimize the neural style loss.
        # This is our initial state: the target image.
        # Note that `scipy.optimize.fmin_l_bfgs_b` can only process flat vectors.
        for i in range(iterations):
            print('Start of iteration', i)
            start_time = time.time()
            self.generated_img, min_val, info = so.fmin_l_bfgs_b(evaluator.loss, self.generated_img, fprime=evaluator.grads, maxfun=20)
            print('Current loss value:', min_val)
            # Save current generated image
            img = self.generated_img.copy().reshape((self.img_height, self.img_width, 3))
            img = self.deprocess_image(img)
            fname = result_prefix + '_at_iteration_%d.png' % i
            imsave(fname, img)
            end_time = time.time()
            print('Image saved as', fname)
            print('Iteration %d completed in %ds' % (i, end_time - start_time))

    def display_final_results(self):
        # Content image
        plt.pyplot.imshow(load_img(self.target_image_path, target_size=(self.img_height, self.img_width)))
        plt.figure()

        # Style im.age
        plt.pyplot.imshow(load_img(self.style_reference_image_path, target_size=(self.img_height, self.img_width)))
        plt.figure()

        # Generate image
        plt.pyplot.imshow(self.generated_img)
        plt.show()


if __name__ == '__main__':
    target_image_path = r'Y:\Documents\StyleNetwork\City_1.jpg'
    style_reference_image_path = r'Y:\Documents\StyleNetwork\Graphic_1.jpg'
    start_image_path = r'Y:\Documents\StyleNetwork\Space_1.jpg'
    stylenet = StyleNetwork(target_image_path, style_reference_image_path, start_image_path)
    stylenet.generate_image()
    stylenet.display_final_results()
