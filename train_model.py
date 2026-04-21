import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

def build_3class_cnn(input_shape=(128, 128, 3)):
    """
    Builds a 3-class Convolutional Neural Network for Brain Tumor Detection.
    Classes: 0 = Normal, 1 = Cancer, 2 = Malformed
    """
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        
        # Final layer: Softmax activation with 3 nodes for 3 classes
        Dense(3, activation='softmax')
    ])
    
    # Compile with categorical_crossentropy
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    model = build_3class_cnn()
    model.summary()
    
    # Example training pseudo-code:
    # model.fit(train_dataset, validation_data=val_dataset, epochs=20)
    # model.save("app/brainTumor.keras")
