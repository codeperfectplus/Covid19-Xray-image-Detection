# -*- coding: utf-8 -*-
'''
@author: codeperfectplus
@github: https://github.com/codeperfectplus

Topic: X-ray Image classification for Covid-19 for Using transfer learning approach(EfficientNetB7)
'''

from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import efficientnet
from tensorflow.keras import layers
from matplotlib import pyplot as plt

IMG_SIZE = 800

train_dir = Path("Covid19-dataset/train")
test_dir = Path("Covid19-dataset/test")

data_gen = ImageDataGenerator(
    preprocessing_function=efficientnet.preprocess_input,
    validation_split=0.2)

train_data = data_gen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb",
    batch_size=1,
    subset="training",
    shuffle=True
)

val_data = data_gen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb",
    batch_size=1,
    subset="validation",
    shuffle=True
)

base_model = efficientnet.EfficientNetB7(include_top=False, weights='imagenet')
base_model.trainable = True

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)
final_output = layers.Dense(3, activation='softmax')(x)

c19_model = keras.models.Model(inputs, final_output)

c19_model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer="adam",
    metrics=['accuracy'])

modelcheckpoint = tf.keras.callbacks.ModelCheckpoint(
    "tmp/saved_model.h5",
    monitor="val_loss",
    verbose=0,
    save_best_only=True,
    save_weights_only=False,
    mode="auto",
    save_freq="epoch",
    options=None)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=5,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=False)

csv_logger = tf.keras.callbacks.CSVLogger(
    "log.csv", separator=",", append=True)


try:
    c19_model.load_weights("tmp/saved_model.h5")
    print("Loading weight from last checkpoint")
except Exception as e:
    print("no weight found. Training from scratch")


c19_model.fit(
    train_data,
    epochs=15,
    batch_size=1,
    validation_data=val_data,
    callbacks=[modelcheckpoint, early_stopping, csv_logger])

# [INFO] plotting the Accracy and loss graph
his = c19_model.history.history

plt.plot(his["accuracy"])
plt.plot(his["val_accuracy"])
plt.legend(["Accuracy", "Val Accuracy"])
plt.savefig("graph/accuracy_Graph.png")
plt.show()

plt.plot(his["loss"])
plt.plot(his["val_loss"])
plt.legend(["Loss", "Val Loss"])
plt.savefig("graph/loss_Graph.png")
plt.show()
