import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from keras import models, layers
import hashlib
import os
import numpy as np

path_to_models = Path('./models')
path_to_csv = Path('./models/models.csv')
training_count = 10
path_to_data_set = Path('./dataset')


class TrainingConfig:
    def __init__(self):
        self.grayscale = False
        self.data_augmentation_mode = "one",
        self.batch_size = 24
        self.img_width = 50
        self.img_height = 50
        self.img_depth = 3
        self.epochs = 15
        self.model_layers = [
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50,50, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(4)
        ]
        self.data_augmentation_methods = []


def train(config):
    # Configuration
    batch_size = config.batch_size
    img_width = config.img_width
    img_height = config.img_height
    epochs = config.epochs
    model_layers = config.model_layers
    grayscale = config.grayscale
    data_augmentation_mode = config.data_augmentation_mode
    data_augmentation_methods = config.data_augmentation_methods

    create_models_folder()

    dateset = tf.keras.utils.image_dataset_from_directory(
        path_to_data_set,
        labels='inferred',
        image_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode='grayscale' if grayscale is True else "rgb",
        shuffle=True,
    )

    train_ds, val_ds, test_ds = split_ds(dateset)

    train_size = len(np.concatenate([i for x, i in train_ds], axis=0))
    val_size = len(np.concatenate([i for x, i in val_ds], axis=0))
    test_size = len(np.concatenate([i for x, i in test_ds], axis=0))

    scale_to_1 = tf.keras.layers.Rescaling(scale=1. / 255)
    scale_to_255 = tf.keras.layers.Rescaling(scale=255 / 1.)

    def augment_data(image, label):
        if data_augmentation_mode is None:
            print("None")
            return image, label

        image = scale_to_1(image)

        if data_augmentation_mode == "one":
            if "brightness" in data_augmentation_methods:
                image = tf.image.random_brightness(image, 0.2)  # brightness
            if "contrast" in data_augmentation_methods:
                image = tf.image.random_contrast(image, 0.5, 2.0)  # contrast
            if "saturation" in data_augmentation_methods:
                image = tf.image.random_saturation(image, 0.75, 1.25)  # saturation
            if "hue" in data_augmentation_methods:
                image = tf.image.random_hue(image, 0.1)  # hue
            if "flip_left_right" in data_augmentation_methods:
                image = tf.image.random_flip_left_right(image)  # flip

            image = scale_to_255(image)
            return image, label
        elif data_augmentation_mode == "multiple":
            images = [scale_to_255(image)]

            if "brightness" in data_augmentation_methods:
                images.append(scale_to_255(tf.image.random_brightness(image, 0.2)))
            if "contrast" in data_augmentation_methods:
                images.append(scale_to_255(tf.image.random_contrast(image, 0.5, 2.0)))
            if "saturation" in data_augmentation_methods:
                images.append(scale_to_255(tf.image.random_saturation(image, 0.75, 1.25)))
            if "hue" in data_augmentation_methods:
                images.append(scale_to_255(tf.image.random_hue(image, 0.1)))
            if "flip_left_right" in data_augmentation_methods:
                images.append(scale_to_255(tf.image.random_flip_left_right(image)))

            labels = tf.repeat(label, repeats=len(data_augmentation_methods) + 1)

            return images, labels

    train_ds = train_ds.map(augment_data)
    if data_augmentation_mode == "multiple":
        train_ds = train_ds.unbatch()

    # Create model
    model = models.Sequential(model_layers, name="Model")
    model_data = (model, epochs, batch_size, img_width, img_height, grayscale, data_augmentation_mode, data_augmentation_methods, train_size, val_size, test_size)
    (model_hash, model_summary) = create_hash(model_data)
    model_path = os.path.join(path_to_models, model_hash)

    print(model_hash)
    print(model_summary)

    # Check if model already exists
    if os.path.exists(model_path):
        print("Model already exists, aborting")
        return

    accuracies = []
    losses = []
    test_accuracies = []

    best = {
        "test_acc": 0,
        "history": None,
        "model": None,
        "index": None
    }

    for i in range(training_count):
        print(f"Training: {i+1} / {training_count}")

        current_model = models.clone_model(model)

        # Train model
        current_model.compile(optimizer='adam',
                              loss=tf.keras.training_losses.SparseCategoricalCrossentropy(from_logits=True),
                              metrics=['accuracy'])

        monitor = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0.001,
            patience=7,
            verbose=0,
            restore_best_weights=True,
            mode='max'
        )

        history = current_model.fit(
            train_ds,
            validation_data=val_ds,
            callbacks=[monitor],
            epochs=epochs,
            verbose=1 if i == 0 else 0,
        )

        predictions = current_model.predict(test_ds)
        correct_predictions = 0

        for image_idx, (x, y) in enumerate(test_ds.unbatch()):
            img_class = y.numpy()
            prediction = np.argmax(predictions[image_idx])

            if prediction == img_class:
                correct_predictions += 1

        test_acc = correct_predictions / len(predictions)
        acc = history.history['accuracy']
        loss = history.history['loss']
        last_acc = acc[-1]
        accuracies.append(last_acc)
        losses.append(loss[-1])
        test_accuracies.append(test_acc)

        if test_acc > best["test_acc"]:
            best["test_acc"] = test_acc
            best["model"] = current_model
            best["history"] = history
            best["index"] = i + 1

    # Create model directory
    os.mkdir(os.path.join(path_to_models, model_hash))

    # Create plots
    create_model_plots(best["history"])
    plt.savefig(os.path.join(model_path, 'results.png'))
    plt.close('all')

    create_training_plot(accuracies, losses, test_accuracies)
    plt.savefig(os.path.join(model_path, 'trainings.png'))
    plt.close('all')

    # Save model data
    best["model"].save(os.path.join(model_path, 'model'))
    file = open(os.path.join(model_path, 'summary.txt'), "w")
    file.write(model_summary)
    file.close()

    # Add model to csv
    final_acc = best["history"].history["accuracy"][-1]
    final_loss = best["history"].history["loss"][-1]
    final_test_acc = best["test_acc"]
    add_to_csv(model_hash, model_data, accuracies, losses, test_accuracies, final_acc, final_loss, final_test_acc)
    print(f"Saved model (run #{best['index']}) with test acc {final_test_acc}, acc {final_acc} and loss {final_loss}")


def split_ds(data_set, train_split=0.8, val_split=0.1, test_split=0.1):
    assert (train_split + test_split + val_split) == 1

    data_set_size = sum(1 for _ in data_set)
    train_size = int(train_split * data_set_size)
    val_size = int(val_split * data_set_size)

    train_ds = data_set.take(train_size)
    val_ds = data_set.skip(train_size).take(val_size)
    test_ds = data_set.skip(train_size).skip(val_size)
    return train_ds, val_ds, test_ds


def create_hash(model_data):
    # Data to save:
    # Batch Size
    # Image Size
    # Epochs
    # Layers
    # Grayscale
    # Data Augmentation Mode
    # Data Augmentation Methods
    # Training Size
    # Validation Size
    # Test Size
    (model, epochs, batch_size, img_width, img_height, grayscale, data_augmentation_mode, data_augmentation_methods, train_size, val_size, test_size) = model_data

    string_list = []
    model.summary(print_fn=lambda x: string_list.append(x))
    string_list.append(f"Batch size: {batch_size}")
    string_list.append(f"Img Width: {img_width}")
    string_list.append(f"Img Height: {img_height}")
    string_list.append(f"Epochs: {epochs}")
    string_list.append(f"Grayscale: {grayscale}")
    string_list.append(f"Data Augmentation Mode: {data_augmentation_mode}")
    string_list.append(f"Data Augmentation Methods: {data_augmentation_methods}")
    string_list.append(f"Train Size: {train_size}")
    string_list.append(f"Val Size: {val_size}")
    string_list.append(f"Test Size: {test_size}")

    model_summary = "\n".join(string_list)

    sha256 = hashlib.sha256(model_summary.encode()).hexdigest()
    return sha256, model_summary


def create_model_plots(history):
    # Get model data
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))  # because of early stopping the number of epochs can be lower than <epochs>

    # Create model plots
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')


def create_training_plot(accuracies, losses, test_accuracies):
    training_range = range(training_count)

    # Create model plots
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 3, 1)
    plt.bar(training_range, accuracies)
    plt.title('Training Accuracies')

    plt.subplot(1, 3, 2)
    plt.bar(training_range, losses)
    plt.title("Training Losses")

    plt.subplot(1, 3, 3)
    plt.bar(training_range, test_accuracies)
    plt.title("Test Accuracies")


def add_to_csv(model_hash, model_data, accuracies, losses, test_accuracies, final_acc, final_loss, test_acc):
    (model, epochs, batch_size, img_width, img_height, grayscale, data_augmentation_mode, data_augmentation_methods, train_size, val_size, test_size) = model_data

    layer_names = [layer.name for layer in model.layers]

    row = [model_hash,
           layer_names,
           len(layer_names),
           img_width,
           img_height,
           epochs,
           batch_size,
           grayscale,
           data_augmentation_mode,
           data_augmentation_methods,
           train_size,
           val_size,
           test_size,
           training_count,
           final_acc,
           final_loss,
           test_acc,
           np.mean(accuracies),
           np.mean(losses),
           np.mean(test_accuracies)]
    row = map(lambda x: str(x), row)
    csv = open(path_to_csv, 'a')
    csv.write(';'.join(row))
    csv.write("\n")
    csv.close()


def create_models_folder():
    if not os.path.exists(path_to_models):
        os.mkdir(path_to_models)

    headers = ["id",
               "layers",
               "layers_count",
               "img_width",
               "img_height",
               "epochs",
               "batch_size",
               "grayscale",
               "data_augmentation_mode",
               "data_augmentation_methods",
               "train_size",
               "val_size",
               "test_size",
               "training_count",
               "acc",
               "loss",
               "test_acc",
               "avg_acc",
               "avg_loss",
               "avg_test_acc"
               ]

    if not os.path.exists(path_to_csv):
        csv = open(path_to_csv, "w")
        csv.write(";".join(headers))
        csv.write("\n")
        csv.close()





