# coding: utf-8
from keras.preprocessing.image import load_img, img_to_array
from keras import backend as K
from keras.applications import vgg19
import scipy.optimize as so
from scipy.misc import imsave
from PIL.ImageEnhance import Sharpness
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
import time
import os


class StyleNetwork:
    def __init__(self, content_img_path, style_img_path, starting_img_path=None, use_mask=None):
        self.content_img_path = content_img_path
        self.style_img_path = style_img_path
        self.final_img_path = None
        width, height = load_img(content_img_path).size

        self.img_height = 400
        self.img_width = int(width * self.img_height / height)
        self.use_mask = use_mask
        if starting_img_path is not None:
            self.generated_img = self.preprocess_image(starting_img_path, normalize=use_mask)
        else:
            self.generated_img = np.random.uniform(0, 255, (1, self.img_height, self.img_width, 3)) - 128.
        self.generated_img = self.generated_img.flatten()

        self.combination_image = K.placeholder((1, self.img_height, self.img_width, 3))
        self.model_layers = self._create_tensor()
        self.style_losses = []
        self.content_losses = []
        self.loss_values = []
        self.loss_changes = []

    def preprocess_image(self, image_path, normalize=None):
        img = load_img(image_path, target_size=(self.img_height, self.img_width))
        img = img_to_array(img)
        if normalize == 'greyscale':
            img = self.black_white_img(img)
        elif normalize == 'mean':
            img = self.img_mean(img)

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

    def img_mean(self, img):
        for row in range(len(img)):
            for pixel in range(len(img[row])):
                row_mean = np.mean(img[row][pixel])
                img[row][pixel] = np.array([row_mean, row_mean, row_mean])
        return img

    def img_sharpen(self, img, factor=2.0):
        img = Image.fromarray(img, 'RGB')
        enhancer = Sharpness(img)
        img = enhancer.enhance(factor)
        img = img_to_array(img)
        return img

    def black_white_img(self, img):
        for row in range(len(img)):
            for pixel in range(len(img[row])):
                row_mean = np.mean(img[row][pixel])
                if row_mean <= 32:
                    img[row][pixel] = np.array([0, 0, 0])
                elif 32 < row_mean <= 64:
                    img[row][pixel] = np.array([32, 32, 32])
                elif 64 < row_mean <= 96:
                    img[row][pixel] = np.array([64, 64, 64])
                elif 96 < row_mean <= 128:
                    img[row][pixel] = np.array([96, 96, 96])
                elif 128 < row_mean <= 160:
                    img[row][pixel] = np.array([128, 128, 128])
                elif 160 < row_mean <= 192:
                    img[row][pixel] = np.array([160, 160, 160])
                elif 192 < row_mean <= 224:
                    img[row][pixel] = np.array([192, 192, 192])
                elif 224 < row_mean <= 230:
                    img[row][pixel] = np.array([224, 224, 224])
                else:
                    img[row][pixel] = np.array([255, 255, 255])
        return img

    def content_loss(self, base, combination):
        return 0.5 * K.sum(K.square(combination - base))

    def gram_matrix(self, x):
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
        gram = K.dot(features, K.transpose(features))
        return gram

    def style_loss(self, style, combination):
        S = self.gram_matrix(style)
        C = self.gram_matrix(combination)
        channels = 3
        size = self.img_height * self.img_width
        return 0.5 * K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

    def total_variation_loss(self, x):
        a = K.square(x[:, :self.img_height - 1, :self.img_width - 1, :] - x[:, 1:, :self.img_width - 1, :])
        b = K.square(x[:, :self.img_height - 1, :self.img_width - 1, :] - x[:, :self.img_height - 1, 1:, :])
        return 0.5 * K.sum(K.pow(a + b, 1.25))

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

    def generate_image(self, content_layer='block3_conv4', style_weight=1., content_weight=0.05, total_variation_weight=1e-4, sharpen=2.0):
        # Name of layer used for content loss
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
            # self.style_losses.append(sl)
            loss += (style_weight / len(style_layers)) * sl
            # self.content_losses.append(loss)

        loss += total_variation_weight * self.total_variation_loss(self.combination_image)
        # self.content_losses.append(np.mean(loss))
        grads = K.gradients(loss, self.combination_image)[0]
        fetch_loss_and_grads = K.function([self.combination_image], [loss, grads])

        evaluator = self.Evaluator(fetch_loss_and_grads, self.img_height, self.img_width)

        self.output_dir = '.\Results City from Dog {}-{} {}{}'.format(style_weight, content_weight, content_layer, ('sharpen' + str(sharpen) if sharpen is not None else ''))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        result_prefix = self.output_dir + r'\style_transfer_result'
        iterations = 60


        img = self.generated_img.copy().reshape((self.img_height, self.img_width, 3))
        img = self.deprocess_image(img)
        img = self.img_sharpen(img, sharpen)
        fname = result_prefix + '_at_iteration_0.png'
        imsave(fname, img)
        print('Image saved as', fname)

        # Run scipy-based optimization (L-BFGS) over the pixels of the generated image
        # so as to minimize the neural style loss.
        # This is our initial state: the target image.
        # Note that `scipy.optimize.fmin_l_bfgs_b` can only process flat vectors.
        min_val = 0
        for i in range(iterations):
            print('Start of iteration', (i + 1))
            start_time = time.time()
            last_min = min_val
            self.generated_img, min_val, info = so.fmin_l_bfgs_b(evaluator.loss, self.generated_img, fprime=evaluator.grads, maxfun=20)
            if last_min == 0:
                loss_change = None
            else:
                loss_change = last_min - min_val
                self.loss_changes.append(loss_change)
            print('Current loss value:', min_val)
            print('Loss amount:', loss_change)
            self.loss_values.append(min_val)
            # Save current generated image
            img = self.generated_img.copy().reshape((self.img_height, self.img_width, 3))
            img = self.deprocess_image(img)
            img = self.img_sharpen(img, sharpen)
            fname = result_prefix + '_at_iteration_%d.png' % (i + 1)
            imsave(fname, img)
            end_time = time.time()
            print('Image saved as', fname)
            print('Iteration %d completed in %ds' % (i, end_time - start_time))
            if loss_change is not None and loss_change < 50000000:
                print('early stopping')
                break
        self.final_img_path = fname

    def display_final_results(self):
        # Content image
        plt.figure(1)

        plt.subplot(131)
        plt.imshow(load_img(self.content_img_path, target_size=(self.img_height, self.img_width)))
        plt.axis('off')
        plt.title('Content Image')

        # Style image
        plt.subplot(132)
        plt.imshow(load_img(self.style_img_path, target_size=(self.img_height, self.img_width)))
        plt.axis('off')
        plt.title('Style Image')

        # Generated image
        plt.subplot(133)
        plt.imshow(load_img(self.final_img_path, target_size=(self.img_height, self.img_width)))
        plt.axis('off')
        plt.title('Generated Image')
        plt.show()


        plt.figure(2)

        # Loss Values
        plt.subplot(121)
        plt.plot(np.arange(len(self.loss_values)), np.array(self.loss_values))
        plt.yscale('log')
        plt.title('Loss Values')

        # Loss Change
        plt.subplot(122)
        plt.plot(np.arange(len(self.loss_changes)), np.array(self.loss_changes))
        plt.yscale('log')
        plt.title('Loss Change')

        plt.gca().yaxis.set_minor_formatter(NullFormatter())
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
        plt.savefig(self.output_dir + r'\loss_plots.png', bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    target_image_path = r'Y:\Documents\StyleNetwork\City_1.jpg'
    style_reference_image_path = r'Y:\Documents\StyleNetwork\Graphic_1.jpg'
    start_image_path = r'Y:\Documents\StyleNetwork\City_1.jpg'
    stylenet = StyleNetwork(target_image_path, style_reference_image_path, start_image_path, use_mask='mean')
    stylenet.generate_image(content_layer='block3_conv4')
    stylenet.display_final_results()
