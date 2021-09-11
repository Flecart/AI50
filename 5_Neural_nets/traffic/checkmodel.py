import numpy as np
import sys
import tensorflow as tf
import cv2
import os

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    
    # Check command-line arguments
    if len(sys.argv) != 3:
        sys.exit("Usage: python checkmodel.py data_dir model")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = tf.keras.models.load_model(sys.argv[2])
    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []

    

    for i in range(NUM_CATEGORIES):
        current_path = os.path.join(data_dir, str(i))

        # f is the name of the file in the path i'm currently looking at
        for f in os.listdir(current_path):
            image_path = os.path.join(current_path, f)

            # reading and resizing cv2 read image
            img = cv2.imread(image_path)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

            images.append(img)
            labels.append(i)
    return (images, labels)

if __name__ == "__main__":
    main()
