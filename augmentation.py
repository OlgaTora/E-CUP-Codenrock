import tensorflow as tf


def process_image(image, label, img_size):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.stateless_random_brightness(
      image, max_delta=0.95, seed=(1,2))
    return image, label
