from PyQt5.Qt import QWidget, QPushButton, QLabel, QVBoxLayout, QPixmap, QFileDialog, QApplication
from keras.preprocessing import *
from PIL import Image
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

def predict(filename):

    tf.random.set_seed(1)

    img_dir = "D:/Учёба/8 семестр СПбГУ/проектирование/3 задание/images/images/images"

    rescale = 1.0 / 255
    IMG_SIZE = 224
    TARGET_SIZE = (IMG_SIZE, IMG_SIZE)
    CLASSES = os.listdir(img_dir)
    BATCH_SIZE = 64

    train_batch_size = BATCH_SIZE

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

    model = tf.keras.models.load_model('D:/model')

    train_input_shape = (224, 224, 3)
    test_image = image.load_img(filename, target_size=(train_input_shape[0:2]))

    test_image = image.img_to_array(test_image)
    test_image /= 255.
    test_image = np.expand_dims(test_image, axis=0)

    labels = train_generator.class_indices
    labels = dict((v, k) for k, v in labels.items())
    prediction = model.predict(test_image)
    prediction_idx = np.argmax(prediction)
    return labels[prediction_idx + 1].replace('_', ' ')

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.button_open = QPushButton('Выбрать картинку')
        self.button_open.clicked.connect(self._on_open_image)

        self.button_save_as = QPushButton('Старт')
        self.button_save_as.clicked.connect(self.predict_image)

        self.label_image = QLabel()
        self.label_text = QLabel('', self)
        self.setGeometry(300, 300, 500, 500)
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.button_open)
        main_layout.addWidget(self.button_save_as)
        main_layout.addWidget(self.label_text)
        main_layout.addWidget(self.label_image)

        self.setLayout(main_layout)

    def _on_open_image(self):
        self.file_name = QFileDialog.getOpenFileName(self, "Выбор картинки", None, "Image (*.png *.jpg)")[0]
        if not self.file_name:
            return

        pixmap = QPixmap(self.file_name)
        self.label_image.setPixmap(pixmap)

    def predict_image(self):
        result = predict(self.file_name)
        self.label_text.setText('Result: ' + result)


if __name__ == '__main__':
    app = QApplication([])

    mw = MainWindow()
    mw.show()

    app.exec()
