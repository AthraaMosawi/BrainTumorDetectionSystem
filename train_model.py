import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input

def build_multimodal_resnet_fusion(input_shape=(224, 224, 3)):
    """
    State-of-the-Art Early Fusion Multi-Modal Network.
    Channel 0 (R) = MRI (Soft tissue)
    Channel 1 (G) = X-Ray (Anatomical bone structure)
    Channel 2 (B) = MWI (Dielectric hotspots)
    Classes: 0 = Normal, 1 = Cancer, 2 = Malformed
    """
    # Load ResNet50 pre-trained on ImageNet (excluding the final classification layer)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze the base model to retain core feature extraction (like edge detection)
    # In a real scenario, you'd unfreeze the top few layers for fine-tuning
    base_model.trainable = False 
    
    # Custom classifier head for our 3 specific medical classes
    x = base_model.output
    x = GlobalAveragePooling2D()(x) # Better than Flatten() for ResNet
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Final 3-node softmax layer for Normal, Cancer, Malformed
    predictions = Dense(3, activation='softmax')(x)
    
    # Construct the final model
    fusion_model = Model(inputs=base_model.input, outputs=predictions)
    
    fusion_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return fusion_model

if __name__ == "__main__":
    adv_model = build_multimodal_resnet_fusion()
    adv_model.summary()
    
    # Example training pseudo-code:
    # adv_model.fit(train_dataset, validation_data=val_dataset, epochs=20)
    # adv_model.save("app/brainTumor.keras")
