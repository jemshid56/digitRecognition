import tensorflow as tf
from tensorflow import keras
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from keras.layers import Dense, Flatten
from keras import Input
from tensorflow.keras.datasets import mnist, fashion_mnist
import argparse
import yaml
from datetime import datetime

def train(config_path):
	with open(config_path) as fh:
		config = yaml.safe_load(fh)
	print(config)

	model_file = config["model_dir"]+"/"+"mnist_model_func.h5"
	buildtime_file = "report"+"/"+"buildtime"
	print(model_file)
	(train_images, train_labels),(test_images, test_labels) = mnist.load_data()

	print(f"train_images.shape = {train_images.shape}")
	print(f"train_labels.shape = {train_labels.shape}")

	train_images = train_images.reshape((len(train_images),
                                     28*28)).astype("float32")/255
	test_images = test_images.reshape((len(test_images),
                                     28*28)).astype("float32")/255

	inputs = Input(shape=(784), name="input_layer")
	features = Dense(64, activation="relu",name="first_layer")(inputs) #f(inputs)
	outputs = Dense(10, activation="softmax", name="output_layer")(features) #f(features)

	fun_model = keras.Model(inputs,outputs)
	fun_model.compile(optimizer =keras.optimizers.Adam(),loss = keras.losses.SparseCategoricalCrossentropy(), metrics = ["accuracy"])
	fun_model.fit(train_images,train_labels, epochs = 4)
	fun_model.save(model_file)

	with open(buildtime_file, 'a') as file:
		file.write('Build Time Printed string Recorded at: %s\n' %datetime.now())
		file.close()

if __name__=="__main__":
	args = argparse.ArgumentParser()
	args.add_argument("--config", default="params.yaml")
	parsed_args = args.parse_args()
	train(config_path = parsed_args.config)
