## Summary
I made 5 numerated attempts and 2 attempts with brute force.
The numerated attempts correspond with the model.try-number, e.g. attempt 5 correspond to model.try5, 4 to model.try4 etc.

The best i could get was with model5 and 95% accuracy, i think its pretty good.

### note
if i didn't rename this file .py submit50 would ignore this
## Attempts

### Attempt 1
The first attempt i made was the easiest one: just copy pasting the network i found in the notes, so i pasted this
```python
model = tf.keras.models.Sequential([ 
    # Convolutional layer. Learn 32 filters using a 3x3 kernel
    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu", input_shape=(30, 30, 3)
    ),
    # Max-pooling layer, using 2x2 pool size
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    # Flatten units
    tf.keras.layers.Flatten(),
    # Add a hidden layer with dropout
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    # Add an output layer with output units for all 10 digits
    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
])
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
```
And got after 10 epochs 333/333loss: 1.5782 - accuracy: 0.5494, but it took about a whole minute to train and test, so from now on i'm switching to the smaller dataset

Maybe the smaller dataset is too small and so it's imprecise.
I used the same network on the smaller dataset, run it multiple times and got these results:
11/11 - 0s - loss: 0.0087 - accuracy: 1.0000
11/11 - 0s - loss: 0.0055 - accuracy: 0.9970
11/11 - 0s - loss: 4.8177e-06 - accuracy: 1.0000

so now i think i'm working on the bigger dataset again.
I just run it again on the bigger dataset, i don't know why, maybe its cached, it ran in 3 second with these results:
40/40 - 0s - loss: 0.1975 - accuracy: 0.9127
Oh i forgot to change the NUM_categories back to 43.
I just ran it again and got this result 
333/333 - 1s - loss: 3.5025 - accuracy: 0.0543
333/333 - 1s - loss: 3.4928 - accuracy: 0.0554

It's perplexing to see that there is a 50% difference with the first try
i did one with 30 epocs and the result is still not so good, i don understand how the first one got 55%
333/333 - 1s - loss: 3.5035 - accuracy: 0.0555


### Attempt 2
I'm trying  new stuff and see which one works better:

```python
model = tf.keras.models.Sequential([ 
        # Convolutional layer. Learn 32 filters using a 3x3 kernel
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(30, 30, 3)
        ),

        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),


        tf.keras.layers.Conv2D(
            16, (3, 3), activation="relu", input_shape=(30, 30, 3)
        ),

        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # Flatten units
        tf.keras.layers.Flatten(),

        # Add a hidden layer with dropout
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        # Add an output layer with output units for all 10 digits
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
```
adding new convolutional and pooling layers does not work: 
333/333 - 1s - loss: 3.5010 - accuracy: 0.0501


### Attempt optimizers
I tried to change the optimizers,i made a loop with all optimizers available

Adam - Fitting model with adam took 48.594635009765625 seconds, the fit is like the one up there
changing the main in this way: 
```python

optimizers = ['adam', 'adadelta', 'adagrad', 'adamax', 'ftrl', 'nadam', 'optimizer', 'rmsprop', "sgd"]
def main():
    # ... [TRUCATED]


    # Get a compiled neural network
    for opt in optimizers:
        print(f"Currently looking at { opt } optimizer")
        a = time.time()
        model = get_model(opt)

        # Fit model on training data
        model.fit(x_train, y_train, epochs=EPOCHS)

        # Evaluate neural network performance
        model.evaluate(x_test,  y_test, verbose=2)

        # Save model to file
        if len(sys.argv) == 3:
            filename = sys.argv[2] + opt
            model.save(filename)
            print(f"Model saved to {filename}.")

        print(f"Fitting model with { opt } took {time.time() - a} seconds")

def get_model(optimizer):
    # ... [TRUCATED]

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
```

But the results i got where similiar to before.
Plus when i'm running the model the CPU goes 100%, i should switch to GPU.

### Attempt 3
333/333 - 1s - loss: 1.6902 - accuracy: 0.4160
This is not so bad result, i tried to mix some activation functions with those, maybe i can do something else.
```python
def get_model():
    model = tf.keras.models.Sequential([ 
        # Convolutional layer. Learn 32 filters using a 3x3 kernel
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(30, 30, 3)
        ),

        tf.keras.layers.MaxPooling2D(pool_size=(5,5)),


        tf.keras.layers.Conv2D(
            16, (3, 3), activation="sigmoid", input_shape=(30, 30, 3)
        ),

        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # Flatten units
        tf.keras.layers.Flatten(),

        # Add a hidden layer with dropout
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        # Add an output layer with output units for all 10 digits
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
```

I run this attempt again, don't know how to load the model, but why this time i get these results?
333/333 - 1s - loss: 3.4820 - accuracy: 0.0653

it is the same model, i don't understand why it's different.
i Just looked how to load model using example here https://cdn.cs50.net/ai/2020/spring/lectures/5/src5/digits/recognition.py
I loaded model.try3 and indeed its 41.6% accuracy, don't know why though.


### Attempt activations
I made a test in the activations, like this:
```python
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
            model.fit(x_train, y_train, epochs=EPOCHS)

            # Evaluate neural network performance
            model.evaluate(x_test,  y_test, verbose=2)

            # Save model to file
            if len(sys.argv) == 3:
                filename = sys.argv[2] + f"{act1}-{act2}"
                model.save(filename)
                print(f"Model saved to {filename}.")

            print(f"Fitting model with { act1 } with { act2 } took {time.time() - a} seconds")


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

```

I did this with the smaller dataset, you can see it in activation folder.
There are many tries that have 100% accuracy, probably because its just a small dataset.
I submit the logs, but didn't submit the models because they where too many


### Attempt 4
i got 88% accuracy!!
I just doubled the size of the dense layers
333/333 - 1s - loss: 0.4835 - accuracy: 0.8838
```python
def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential([ 
        # Convolutional layer. Learn 32 filters using a 3x3 kernel
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(30, 30, 3)
        ),

        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten units
        tf.keras.layers.Flatten(),

        # Add a hidden layer with dropout
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.4),

        # Add an output layer with output units for all 10 digits
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    model.compile(
        optimizer='adam',
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
```

### Attempt 5
333/333 - 2s - loss: 0.2833 - accuracy: 0.9367
93%! I added more dense layers, 300 dense layers and more training batches.
I'm satisfied with the result.
```python
EPOCHS = 20
def get_model():
    # ...[TRUNCATED]
    model = tf.keras.models.Sequential([ 
        # ...[TRUNCATED]
        tf.keras.layers.Dense(300, activation="relu"),
        # ...[TRUNCATED]
    ])
    # ...[TRUNCATED]
    return model
```
I'm making more tests on this with checkmodel.py, seems like it is stabilized on 95% accuracy
333/333 - 2s - loss: 0.1709 - accuracy: 0.9587
333/333 - 1s - loss: 0.1807 - accuracy: 0.9567
333/333 - 1s - loss: 0.1781 - accuracy: 0.9550
