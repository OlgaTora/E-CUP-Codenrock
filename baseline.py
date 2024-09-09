import tensorflow as tf

from functools import partial
from dataset import DatasetCreator
from model import ModelTransferLearning
from augmentation import process_image
from const import IMAGE_SIZE

AUTOTUNE = tf.data.experimental.AUTOTUNE


if __name__ == "__main__":
    dataset = DatasetCreator()
    train_ds, val_ds = dataset.preproccesing()
    train_ds = train_ds.map(partial(process_image, img_size=IMAGE_SIZE[0]))

    train_ds_ = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    model = ModelTransferLearning()
    transfer_model = model.create_model()
    weigths = dataset.get_weights()
    model = model.train_model(transfer_model, train_ds, val_ds, weigths)
    model.save("model.keras")
