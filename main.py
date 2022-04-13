import os

import numpy as np
import pandas as pd

import random

from sklearn import metrics

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, preprocessing, layers, callbacks, optimizers
import sklearn

import matplotlib.pyplot as plt
import seaborn as sns

from numpy.random import seed

seed(1) #Seed function is used to save the state of random function

tf.random.set_seed(1)

img_dir = "D:/Учёба/8 семестр СПбГУ/проектирование/3 задание/images/images/images"
label_df = pd.read_csv("D:/Учёба/8 семестр СПбГУ/проектирование/3 задание/images/artists.csv")

classes = os.listdir(img_dir)

df = label_df
df = df.sort_values(by=['name'], ascending=True)

rescale = 1.0 / 255
IMG_SIZE = 224
TARGET_SIZE = (IMG_SIZE, IMG_SIZE)
CLASSES = os.listdir(img_dir)
NUM_CLASSES = len(CLASSES)
BATCH_SIZE = 64
# BATCH_SIZE = 8 * strategy.num_replicas_in_sync

train_batch_size = BATCH_SIZE
validation_batch_size = BATCH_SIZE * 5
test_batch_size = BATCH_SIZE * 5


# Calculate Class Weights
def get_weight(y, NUM_CLASSES):
    class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight="balanced",
                                         classes=np.unique(y),
                                         y=y
    )
    return dict(enumerate(class_weights))


train_datagen = preprocessing.image.ImageDataGenerator(
    rescale=rescale,
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.05)

train_generator = train_datagen.flow_from_directory(
    img_dir,
    target_size=TARGET_SIZE,
    classes=CLASSES,
    class_mode="categorical",
    batch_size=train_batch_size,
    shuffle=True,
    seed=42,
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    img_dir,
    classes=CLASSES,
    target_size=TARGET_SIZE,
    class_mode="categorical",
    batch_size=validation_batch_size,
    shuffle=False,
    seed=42,
    subset='validation')

class_weights = get_weight(train_generator.classes, NUM_CLASSES)

steps_per_epoch = len(train_generator)
validation_steps = len(validation_generator)

INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

weights = 'imagenet'
dense_units = 1024

inputs = layers.Input(INPUT_SHAPE)

base_model = tf.keras.applications.InceptionResNetV2(
    include_top=False,
    weights=weights,
    input_shape=INPUT_SHAPE,
)

x = base_model.output
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)

x = layers.Dense(dense_units)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation(activation='relu')(x)
x = layers.Dropout(0.5)(x)

outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs=base_model.input, outputs=outputs)

OPTIMIZER = optimizers.Adam()
# OPTIMIZER = optimizers.Adam(learning_rate=0.0001)

EARLY_STOPPING = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1,
    restore_best_weights=True)


REDUCE_LR = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.8,
    patience=1,
    min_lr=0.000001,
    verbose=1)

CALLBACKS = [REDUCE_LR, EARLY_STOPPING]

model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])


VERBOSE = 1
EPOCHS = 100
"""
print("Trainning Model ...\n")
history = model.fit_generator(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    verbose=VERBOSE,
    callbacks=CALLBACKS,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    class_weight=class_weights
)
"""
#model.save('D:/model')

from keras.preprocessing import *

model = tf.keras.models.load_model('D:/model')

from PIL import Image
train_input_shape = (224, 224, 3)
test_image = image.load_img('D:/1.jpg', target_size=(train_input_shape[0:2]))

test_image = image.img_to_array(test_image)
test_image /= 255.
test_image = np.expand_dims(test_image, axis=0)

labels = train_generator.class_indices
labels = dict((v,k) for k,v in labels.items())
prediction = model.predict(test_image)
prediction_idx = np.argmax(prediction)
print("Predicted artist =", labels[prediction_idx+1].replace('_', ' '))
