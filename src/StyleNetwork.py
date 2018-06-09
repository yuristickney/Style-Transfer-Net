from keras.preprocessing.image import load_img, img_to_array
import  numpy  as  np
from keras.applications import  vgg19
from keras import backend as K


class StyleNetwork:
    def __init__(self, content_img_path, style_img_path, content_weight = 1.0, style_weight=0.1, img_height=400):
        self.img_height = img_height
        width, height = load_img(content_img_path).size
        self.img_width = int(width * img_height / height)
        self.input_tensor = self._create_tensor(content_img_path, style_img_path)
        self.model = vgg19.VGG16(input_tensor=self.input_tensor, weights='imagenet', include_top=False)
        print('Init:VGG19 Model loaded.')

        self.model_layers = {layer.name: layer.output for layer in self.model.layers}

        self.content_layer = 'block5_conv2'

        self.style_layers = ['block1_conv1', 'block1_conv2',
                        'block2_conv1', 'block2_conv2',
                        'block3_conv1', 'block3_conv2',
                        'block4_conv1', 'block4_conv2',
                        'block5_conv1', 'block5_conv2']

        self.total_variation_weight = 1e-4
        self.style_weight = style_weight
        self.content_weight = content_weight

        self.loss = K.variable(0.)
        layer_features = self.outputs_dict[self.content_layer]

        content_img_features = layer_features[0, :, :, :]
        comb_features = layer_features[2, :, :, :]
        self.loss += content_weight * self.content_loss(content_img_features, comb_features)


    def _preprocess_img(self, img_path):
        img = load_img(img_path, target_size=(self.img_height, self.img_width))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = vgg19.preprocess_input(img)
        return img

    def _deprocess_image(self, img):
        # Remove zero-center by mean pixel
        img[:, :, 0] += 103.939
        img[:, :, 1] += 116.779
        img[:, :, 2] += 123.68
        # 'BGR'->'RGB'
        img = img[:, :, ::-1]
        img = np.clip(img, 0, 255).astype('uint8') #ensure color values are between 0 and 255
        return img

    def _create_tensor(self, content_img_path, style_img_path):
        content_img = K.constant(self._preprocess_img(content_img_path))
        style_img = K.constant(self._preprocess_img(style_img_path))
        generated_image = K.placeholder((1, self.img_height, self.img_width, 3))  # change to noise image
        return K.concatenate([content_img, style_img, generated_image], axis=0)

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

    def print_layers(self):
        for layer in self.model.layers:
            print(layer.name, layer.output, '\n')

