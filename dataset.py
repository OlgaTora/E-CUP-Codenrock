from PIL import Image
import os
import numpy as np
import shutil
import tensorflow as tf
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder
from const import TRAIN_DIR, BATCH_SIZE, IMAGE_SIZE


class DatasetCreator:

    def __init__(self):
        self.data_dir = TRAIN_DIR
        self.labels = []
        self.image_size = IMAGE_SIZE
        self.batch_size = BATCH_SIZE

    def __get_2classes(self):
        """Move file into 2 directories for get 2 classes from 5 classes"""
        subdirs = ["cigs", "pipes", "roll_cigs"]
        destination = os.path.join(self.data_dir, "smoking")
        for dir in subdirs:
            for file_name in os.listdir(os.path.join(self.data_dir, dir)):
                source = os.path.join(self.data_dir, dir, file_name)
                dest = os.path.join(destination, file_name)
                shutil.move(source, dest)
            os.removedirs(os.path.join(self.data_dir, dir))

    def __convert_file2jpeg(self, file_name, dir):
        """Function for convert files to jpeg format"""
        if file_name.lower().endswith("webp") or file_name.lower().endswith("jpg"):
            img = Image.open(os.path.join(self.data_dir, dir, file_name)).convert("RGB")
            img.save(f"{os.path.join(self.data_dir, dir, file_name[:-4])}.jpeg", "JPEG")
            os.remove(os.path.join(self.data_dir, dir, file_name))

    def __get_labels(self):
        """Function for get labels from dataset"""
        for dir in os.listdir(self.data_dir):
            for file in os.listdir(os.path.join(self.data_dir, dir)):
                self.__convert_file2jpeg(file, dir)
                self.labels.append(dir)
        enc = LabelEncoder()
        self.labels = enc.fit_transform(self.labels)
        return self.labels

    def get_weights(self):
        """Function for get weights for each class"""
        class_weights = class_weight.compute_class_weight(
            "balanced", classes=np.unique(self.labels), y=self.labels
        )
        return dict(enumerate(class_weights))

    def split_data(self):
        train_ds, val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.15,
            subset="both",
            seed=123,
            image_size=self.image_size,
            batch_size=self.batch_size,
            label_mode="categorical",
        )
        return train_ds, val_ds

    def preproccesing(self):
        self.__get_2classes()
        self.labels = self.__get_labels()
        return self.split_data()
