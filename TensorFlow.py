import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds 
tfds.disable_progress_bar()

def format_example(image, label):
    # returns an image that is reshaped to a constant size
    IMG_SIZE = 160
    
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

def graph_data_matrix(dataMatrix):
    plt.figure()
    plt.imshow(dataMatrix)
    plt.colorbar()
    plt.grid(False)
    plt.savefig("figures/2d_layer.png")

def deep_network():
    # load the training and testing data
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # preprocess the data, every value is between 0 and 1
    train_images = train_images / 255.0
    test_images  = test_images / 255.0

    # build the model, has three layers, input, hidden, and output
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    # comiles the neural network
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']) 

    # train the model
    model.fit(train_images, train_labels, epochs=1)

    # test the model
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
    print('Test accuracy: ', test_acc)

    # predict the class of an image
    predictions = model.predict(test_images)
    print(np.argmax(predictions[0])) 
    
def convolutional_network():
    # load and normalize data
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255, test_images / 255
    
    # build convolutional architecture
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    
    # flatten neural network into a dense neural network
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    
    # train the network
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=1,
                        validation_data=(test_images, test_labels))

def pretrained_convolutional_network():
    # split the data into 80% training, 10% testing, 10% validation
    (raw_train, raw_validation, raw_test), metadata = tfds.load(
        'cats_vs_dogs',
        split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
        with_info=True,
        as_supervised=True,
    )
    
    # create a function object that we can user to get labels
    get_label_name = metadata.features['label'].int2str
    
    # uses format_example function to resize all training images
    train = raw_train.map(format_example)
    validation = raw_validation.map(format_example)
    test = raw_test.map(format_example)
    
    # pick a pretrained model
    IMG_SHAPE = (160, 160, 3)
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    
    # freeze the base model so that it doesn't train
    base_model.trainable = False
    
    # instead of flattening the entire feature map, average the 5x5 area of
    # each 2D map, this will be the input to our dense network
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    
    # add the prediction layer that will be one neuron, we can do this
    # because we only have 2 classes
    prediction_layer = keras.layers.Dense(1)
    
    # add all of these layers together
    model = tf.keras.Sequential([
        base_model, 
        global_average_layer, 
        prediction_layer
    ])
    
    # train the network
    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    # train the model
    BATCH_SIZE = 32
    SHUFFLE_BUFFER_SIZE = 1000
    
    train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    validation_batches = validation.batch(BATCH_SIZE)
    test_batches = test.batch(BATCH_SIZE)
    
    history = model.fit(train_batches,
                        epochs=1,
                        validation_data=validation_batches)
    acc = history.history['accuracy']
    print(acc)
    
    # save model with keras-specific file type, .h5
    model.save("dogs_vs_cats.h5")
    new_model = tf.keras.models.load_model("dogs_vs_cats.h5")
    
if __name__ == "__main__":
    pretrained_convolutional_network() 