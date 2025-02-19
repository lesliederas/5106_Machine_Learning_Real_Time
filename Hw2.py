import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
# Problem 1
print("Problem 1 \n")
def load_data(dataset="cifar10"):
    if dataset == "cifar10":
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        num_classes = 10
    else:
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
        num_classes = 100

    # Normalize to [0,1] and convert labels to one-hot encoding
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test, num_classes

# Load CIFAR-10
x_train, y_train, x_test, y_test, num_classes = load_data("cifar10")  # Change to "cifar100" for CIFAR-100

def build_alexnet(input_shape=(32, 32, 3), num_classes=10, use_dropout=False):
    model = keras.Sequential([
        layers.Conv2D(64, (3,3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        
        layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        
        layers.Flatten(),
        
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5 if use_dropout else 0.0),  # Dropout if enabled
        
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5 if use_dropout else 0.0),  # Dropout if enabled
        
        layers.Dense(num_classes, activation='softmax')  # Output layer
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Build models with and without dropout
model_no_dropout = build_alexnet(use_dropout=False)
model_with_dropout = build_alexnet(use_dropout=True)

# Show model summary
model_with_dropout.summary()
def train_and_evaluate(model, x_train, y_train, x_test, y_test, epochs=20, batch_size=64):
    history = model.fit(x_train, y_train, 
                        validation_data=(x_test, y_test), 
                        epochs=epochs, 
                        batch_size=batch_size)
    
    test_loss, test_acc = model.evaluate(x_test, y_test)
    return history, test_loss, test_acc

# Train both models
history_no_dropout, loss_no_dropout, acc_no_dropout = train_and_evaluate(model_no_dropout, x_train, y_train, x_test, y_test)
history_with_dropout, loss_with_dropout, acc_with_dropout = train_and_evaluate(model_with_dropout, x_train, y_train, x_test, y_test)

print(f"Model without Dropout: Test Loss = {loss_no_dropout:.4f}, Test Accuracy = {acc_no_dropout:.4f}")
print(f"Model with Dropout: Test Loss = {loss_with_dropout:.4f}, Test Accuracy = {acc_with_dropout:.4f}")

# Compare number of parameters
params_no_dropout = model_no_dropout.count_params()
params_with_dropout = model_with_dropout.count_params()
print(f"Model without Dropout: {params_no_dropout:,} parameters")
print(f"Model with Dropout: {params_with_dropout:,} parameters")

def plot_history(history, title="Training Progress"):
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(title + " - Accuracy")

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(title + " - Loss")

    plt.show()

# Plot results
plot_history(history_no_dropout, "No Dropout")
plot_history(history_with_dropout, "With Dropout")

print("Problem 2 \n")
#Problem 2
# VGG configurations
vgg_configs = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]
}
def build_vgg(config, input_shape=(32, 32, 3), num_classes=10, use_dropout=False):
    model = keras.Sequential()
    for layer in config:
        if layer == "M":
            model.add(layers.MaxPooling2D((2, 2)))
        else:
            model.add(layers.Conv2D(layer, (3, 3), activation='relu', padding='same'))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5 if use_dropout else 0.0))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate(model, x_train, y_train, x_test, y_test, epochs=20, batch_size=64):
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    return history, test_loss, test_acc

def plot_history(history, title):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(title + " - Accuracy")
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(title + " - Loss")
    plt.show()

# Load CIFAR-10 data
x_train, y_train, x_test, y_test, num_classes = load_data("cifar10")

# Choose the VGG variant closest in parameters to AlexNet
vgg_variant = "VGG13"
model_vgg = build_vgg(vgg_configs[vgg_variant], num_classes=num_classes)

# Train and evaluate VGG
history_vgg, loss_vgg, acc_vgg = train_and_evaluate(model_vgg, x_train, y_train, x_test, y_test)

# Print results
print(f"VGG ({vgg_variant}): Test Loss = {loss_vgg:.4f}, Test Accuracy = {acc_vgg:.4f}")
print(f"VGG ({vgg_variant}) Parameters: {model_vgg.count_params():,}")

# Plot VGG results
plot_history(history_vgg, f"VGG {vgg_variant}")
