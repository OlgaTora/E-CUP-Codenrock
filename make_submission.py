import os
import tensorflow as tf
import numpy as np
from const import MODEL_WEIGHTS, TEST_IMAGES_DIR, SUBMISSION_PATH, IMAGE_SIZE

if __name__ == "__main__":
    model = tf.keras.models.load_model("model.keras")
    model.load_weights(MODEL_WEIGHTS)

    all_image_names = os.listdir(TEST_IMAGES_DIR)
    all_preds = []

    for file in os.listdir(TEST_IMAGES_DIR):
        sample = tf.keras.utils.load_img(
            path=(os.path.join(TEST_IMAGES_DIR, file)),
            target_size=IMAGE_SIZE,
            color_mode="rgb",
        )
        sample = tf.keras.utils.img_to_array(sample)
        sample = tf.expand_dims(sample, 0)
        predictions = model.predict(sample)
        # predictions = np.where(predictions < 0.5, 0, 1)

        predictions = np.argmax(tf.nn.softmax(predictions))
        predictions = tf.where(predictions == 1, 1, 0)
        all_preds.append(int(predictions))

    with open(SUBMISSION_PATH, "w") as f:
        f.write("image_name\tlabel_id\n")
        for name, cl_id in zip(all_image_names, all_preds):
            f.write(f"{name}\t{cl_id}\n")
