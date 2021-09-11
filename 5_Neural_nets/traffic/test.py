import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import time

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 3
TEST_SIZE = 0.4

optimizers = ['adam', 'adadelta', 'adagrad', 'adamax', 'ftrl', 'nadam', 'optimizer', 'rmsprop', "sgd"]
activation = ['relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh', 'selu', 'elu', 'exponential']
def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )


    # Get a compiled neural network
    for act1 in activation:
        for act2 in activation:
            print(f"Currently looking at { act1 } with { act2 }")
            a = time.time()
            model = get_model(activation=act1, second_activation=act2)

            # Fit model on training data
            model.fit(x_train, y_train, epochs=EPOCHS, verbose=0)

            # Evaluate neural network performance
            model.evaluate(x_test,  y_test, verbose=2)

            # Save model to file
            if len(sys.argv) == 3:
                filename = sys.argv[2] + f"{act1}-{act2}"
                model.save(filename)
                print(f"Model saved to {filename}.")

            print(f"Fitting model with { act1 } with { act2 } took {time.time() - a} seconds")


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


def get_model(optimizer="adam", activation="relu", second_activation='relu'):
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential([ 
        # Convolutional layer. Learn 32 filters using a 3x3 kernel
        tf.keras.layers.Conv2D(
            32, (3, 3), activation=activation, input_shape=(30, 30, 3)
        ),

        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten units
        tf.keras.layers.Flatten(),

        # Add a hidden layer with dropout
        tf.keras.layers.Dense(128, activation=second_activation),
        tf.keras.layers.Dropout(0.5),

        # Add an output layer with output units for all 10 digits
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
if __name__ == "__main__":
    main()



# https://stackoverflow.com/questions/26681756/how-to-convert-a-python-numpy-array-to-an-rgb-image-with-opencv-2-4
# I'm trying to understand how cv2 works with this script

# import numpy, cv2
# def show_pic(p):
#         ''' use esc to see the results'''
#         print(type(p))
#         cv2.imshow('Color image', p)
#         while True:
#             k = cv2.waitKey(0) & 0xFF
#             if k == 27: break 
#         return
#         cv2.destroyAllWindows()

# b = numpy.zeros([200,200,3])

# b[:,:,0] = numpy.ones([200,200])*255
# b[:,:,1] = numpy.ones([200,200])*255
# b[:,:,2] = numpy.ones([200,200])*0
# cv2.imwrite('color_img.jpg', b)


# c = cv2.imread('color_img.jpg', 1)
# c = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)

# d = cv2.imread('color_img.jpg', 1)
# d = cv2.cvtColor(c, cv2.COLOR_RGB2BGR)

# e = cv2.imread('color_img.jpg', -1)
# e = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)

# f = cv2.imread('color_img.jpg', -1)
# f = cv2.cvtColor(c, cv2.COLOR_RGB2BGR)


# pictures = [d]

# for p in pictures:
#     show_pic(p)
# # show the matrix
# print(c)
# print(c.shape)


# i don't know why it freezes

# seeing if i have GPU
# import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# I don't have