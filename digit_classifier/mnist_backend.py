import matplotlib
matplotlib.use('TkAgg')  # Set TkAgg backend to try to solve plotting issue
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns

def plot_training_history(history, params):
    """Plots training and validation accuracy and loss curves."""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f"Training and Validation Accuracy\nHyperparameters: {params}")
    plt.xlabel("Epochs")
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f"Training and Validation Loss\nHyperparameters: {params}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()

def create_and_evaluate_mnist_model_with_best_params():
    """
    Creates, trains, and evaluates a CNN model for MNIST digit classification using best hyperparameters.
    """

    # 1. Data Loading and Preprocessing
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    # 2. Define Best Hyperparameters (hard-coded based on prior tuning)
    best_params = {
        'optimizer': 'adam',
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.0005
    }
    print(f"Using best hyperparameters: {best_params}")

    # 3. Build the CNN Model
    model = Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),  # Define input shape explicitly
        Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dropout(0.5),
        Dense(128, activation='relu', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    # 4. Choose Optimizer based on best_params
    lr = best_params['learning_rate']
    if best_params['optimizer'] == 'adam':
        optimizer = Adam(learning_rate=lr)
    elif best_params['optimizer'] == 'rmsprop':
        optimizer = RMSprop(learning_rate=lr)
    else:
        optimizer = Adam(learning_rate=lr)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 5. Train the model using best hyperparameters
    history = model.fit(x_train, y_train,
                        epochs=best_params['epochs'],
                        batch_size=best_params['batch_size'],
                        validation_split=0.1)

    # 6. Evaluate the model on the test set
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Plot training history for visual inspection
    plot_training_history(history, best_params)

    # 7. Detailed Evaluation: Classification Report and Confusion Matrix
    y_pred_prob = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred_prob, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes))

    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.arange(10), yticklabels=np.arange(10))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix - Best MNIST CNN Model')
    plt.show()

    return model, best_params

if __name__ == "__main__":
    best_cnn_model, best_cnn_params = create_and_evaluate_mnist_model_with_best_params()
    print("\nBest Trained CNN model with best parameters is returned and ready for use in frontend.")
    print(f"Best Parameters: {best_cnn_params}")

    # --- Save the best trained model ---
    best_cnn_model.save('mnist_cnn_model.keras')
    print("Best CNN model saved to 'classifier/mnist_cnn_model.keras'")
