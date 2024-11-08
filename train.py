from datetime import datetime

import keras.callbacks
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from model import get_model, pre_trained_model, get_conv_base, pre_trained_model_for_data_augmentation
from preprocessing import load_images


def train_model():
    (train_dataset, validation_dataset, test_dataset) = load_images()
    model = get_model()
    tensorboard = keras.callbacks.TensorBoard(
        log_dir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S"),
        histogram_freq=1
    )
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="convnet_from_scratch_with_augmentation.keras",
            save_best_only=True,
            monitor="val_loss"
        ),
        tensorboard
    ]
    history = model.fit(
        train_dataset,
        epochs=100,
        validation_data=validation_dataset,
        callbacks=callbacks
    )
    return history

def get_features_and_labels(dataset):
    all_features = []
    all_labels = []
    conv_base = get_conv_base()
    for images, labels in dataset:
        preprocessed_images = tf.keras.applications.vgg16.preprocess_input(images)
        features = conv_base.predict(preprocessed_images)
        all_features.append(features)
        all_labels.append(labels)
    return np.concatenate(all_features), np.concatenate(all_labels)


def check_test_accuracy():
    (_, _, test_dataset) = load_images()
    test_model = keras.models.load_model("feature_extraction_with_augmentation.keras")
    test_loss, test_acc = test_model.evaluate(test_dataset)
    print(f"Test accuracy: {test_acc:.3f}")
    return None

def train_preloaded_model():
    (train_dataset, validation_dataset, test_dataset) = load_images()
    train_features, train_labels = get_features_and_labels(train_dataset)
    validation_features, validation_labels = get_features_and_labels(validation_dataset)
    test_features, test_labels = get_features_and_labels(test_dataset)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="feature_extraction.keras",
            save_best_only=True,
            monitor="val_loss"
        )
    ]
    model = pre_trained_model()
    history = model.fit(
        train_features,
        train_labels,
        epochs=20,
        validation_data=(validation_features, validation_labels),
        callbacks=callbacks
    )
    return history

def train_preloaded_model_with_augmentation():
    (train_dataset, validation_dataset, test_dataset) = load_images()
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="feature_extraction_with_augmentation.keras",
            save_best_only=True,
            monitor="val_loss"
        )
    ]
    model = pre_trained_model_for_data_augmentation()
    history = model.fit(
        train_dataset,
        epochs=50,
        validation_data=validation_dataset,
        callbacks=callbacks
    )
    return history

if __name__ == "__main__":
    history = train_preloaded_model_with_augmentation()
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(accuracy) + 1)
    plt.plot(epochs, accuracy, "bo", label="Training accuracy")
    plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    check_test_accuracy()
