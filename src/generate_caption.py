from pickle import load
from numpy import argmax
import argparse
import os
from keras.models import load_model
import cv2
import numpy as np
from PIL import Image
import cv2
import os

def extract_features(filename):
  img_array = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
  img_pil = Image.fromarray(img_array)
  img_28x28 = np.array(img_pil.resize((28, 28), Image.ANTIALIAS))
  img_array = (img_28x28.flatten())
  img_array  = img_array.reshape(-1,1).T/255
  return img_array
 
def generate_captions(photo_path):
        model = load_model('models/mnist_model_func.h5')
        photo = extract_features(photo_path)
        prediction = np.argmax(model.predict(photo.reshape(1,784)))
        return prediction
