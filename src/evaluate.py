import tensorflow as tf
from tensorflow import keras
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten
from keras import Input
from tensorflow.keras.datasets import mnist, fashion_mnist
from keras.models import load_model
import argparse
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, plot_roc_curve,accuracy_score
from sklearn.metrics import average_precision_score
from numpyencoder import NumpyEncoder
import yaml
import json

def evaluate(config_path):
	print(config_path)

    # Convert the YAML data into a dictionary
	with open(config_path) as fh:
    		config = yaml.safe_load(fh)

	print(config)
	(train_images, train_labels),(test_images, test_labels) = mnist.load_data()
	scores_file = config["reports"]["scores"]
	test_images = test_images.reshape((len(test_images), 28*28)).astype("float32")/255
	model = load_model('models/mnist_model_func.h5')
	score = model.evaluate(test_images, test_labels, verbose=0)
	predicted_val = model.predict(test_images);
	print('loss=', score[0])
	print('accuracy=', score[1])

	with open(scores_file, "w+") as f:
		scores = {
			"Loss": score[0],
			"Accuracy": score[1]
		}
		json.dump(scores, f, indent=4)

if __name__=="__main__":
	args = argparse.ArgumentParser()
	args.add_argument("--config", default="params.yaml")
	parsed_args = args.parse_args()
	evaluate(config_path = parsed_args.config)
