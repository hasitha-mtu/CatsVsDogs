import keras.callbacks
import matplotlib.pyplot as plt
from preprocessing import load_images
from model import get_model


def train_model():
    (train_dataset, validation_dataset, test_dataset) = load_images()
    model = get_model()
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="convnet_from_scratch.keras",
            save_best_only=True,
            monitor="val_loss"
        )
    ]
    history = model.fit(
        train_dataset,
        epochs=30,
        validation_data=validation_dataset,
        callbacks=callbacks
    )
    return history

if __name__ == "__main__":
    history = train_model()
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