import tensorflow as tf
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.applications import ResNet50
from pathlib import Path
import gdown
import os

def loadModel():
    # Load the ResNet50 model from Keras applications
    base_model = tf.keras.applications.ResNet50(
        include_top=False,  # Exclude the top layers so we can add custom layers for ArcFace
        input_shape=(112, 112, 3),
        weights=None  # No pretrained weights since ArcFace might require specific tuning
    )
    
    # Define inputs and add custom layers for ArcFace embedding
    inputs = base_model.input
    arcface_model = base_model.output
    arcface_model = layers.BatchNormalization(momentum=0.9, epsilon=2e-5)(arcface_model)
    arcface_model = layers.Dropout(0.4)(arcface_model)
    arcface_model = layers.Flatten()(arcface_model)
    arcface_model = layers.Dense(512, activation=None, use_bias=True, kernel_initializer="glorot_normal")(arcface_model)
    embedding = layers.BatchNormalization(momentum=0.9, epsilon=2e-5, name="embedding", scale=True)(arcface_model)
    
    # Build the model
    model = models.Model(inputs, embedding, name="ArcFaceModel")

    # Workaround: Re-creating the model as a sequential copy
    new_model = Sequential()
    for layer in model.layers:
        new_model.add(layer)
    
    # Define the URL and path for downloading pretrained ArcFace weights if available
    home = str(Path.home())
    url = "https://drive.google.com/uc?id=1LVB3CdVejpmGHM28BpqqkbZP5hDEcdZY"
    file_name = "arcface_weights.h5"
    output = os.path.join(home, '.deepface', 'weights', file_name)
    Path(os.path.dirname(output)).mkdir(parents=True, exist_ok=True)
    
    # Download the weights if they don't exist locally
    if not os.path.isfile(output):
        print(f"{file_name} will be downloaded to {output}")
        gdown.download(url, output, quiet=False)
    
    # Load weights if available
    try:
        new_model.load_weights(output)
        print("Pre-trained weights loaded successfully.")
    except Exception as e:
        print("Pre-trained weights could not be loaded:", e)
        print(f"Please download manually from {url} and place it at {output}.")
    
    return new_model
