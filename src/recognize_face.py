import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau ,  ModelCheckpoint
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt


class InceptionV3FaceRecognitionModel:
    def __init__(self, num_classes, learning_rate=0.001, trainable_base_layers=None):
        """
        Initializes the InceptionV3 model for face recognition with custom layers.

        Args:
            num_classes (int): The number of output classes.
            learning_rate (float): The learning rate for the optimizer (default: 0.001).
            trainable_base_layers (int or None): The number of base model layers to unfreeze (default: None).
        """
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.trainable_base_layers = trainable_base_layers
        self.model = self.build_model()

    def build_model(self):
        """
        Builds and compiles the InceptionV3 model for face recognition.

        Returns:
            model (tf.keras.Model): Compiled Keras model.
        """
        # Load InceptionV3 with pre-trained ImageNet weights, excluding the top layers
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

        # Freeze early layers, unfreeze last `trainable_base_layers` layers
        if self.trainable_base_layers:
            for layer in base_model.layers[:-self.trainable_base_layers]:
                layer.trainable = False
        else:
            base_model.trainable = False

        # Add custom classification layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)  # Reduces spatial dimensions
        x = Dense(1024, activation='relu')(x)
        x = BatchNormalization()(x)  # Normalize activations
        x = Dropout(0.5)(x)  # Regularization

        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)

        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        output_layer = Dense(self.num_classes, activation='softmax')(x)  # Output layer

        # Create model
        model = Model(inputs=base_model.input, outputs=output_layer)

        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def get_model(self):
        """
        Returns the compiled model.

        Returns:
            model (tf.keras.Model): The compiled model.
        """
        return self.model
    

    def train_model(model, train_generator, val_generator, class_weights=None, epochs=20):
        """
        Train the CNN model with early stopping and learning rate reduction.

        Returns:
            history: Training history object.
        """  
        # Callbacks
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        )
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-6
        )

        # Model checkpoint - saves the best model based on val_loss
        checkpoint_callback = ModelCheckpoint(
            filepath=r"best_model.keras",  # Save in HDF5 format
            monitor="val_loss",           # Track validation loss instead of accuracy
            save_best_only=True,          # Save only the best model
            save_weights_only=False,      # Save the entire model, not just weights
            mode="min",                   # Minimize val_loss
            verbose=1                     # Print save status
        )

        # Train the model
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=[early_stopping, reduce_lr, checkpoint_callback],
            class_weight=class_weights,
            verbose=1
        )

        return history
    
    def plot_training_history(self, history):
        """
        Plot the training history (accuracy and loss).
        """
        plt.figure(figsize=(18, 5))

        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()
