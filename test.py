import keras
from keras.preprocessing.image import load_img, img_to_array

import scipy.optimize as optimize
import scipy.misc as sm
import time

import numpy as np
from keras.applications import vgg19
#
from matplotlib import pyplot as plt

from keras import backend as K