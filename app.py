import gradio as gr
import tensorflow as tf
import cv2

model = tf.keras.models.load_model(r"./vgg16_julien.keras")

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def predict(image):
    # Resize image to (32, 32)
    image = cv2.resize(image, (32, 32))
    print("Resized image shape:", image.shape)  # Print the shape of the resized image
    # Convert image to float32 and normalize
    image = image.astype("float32") / 255.0
    # Add batch dimension
    image = tf.expand_dims(image, 0)
    # Predict using the model
    prediction = model.predict(image)
    class_index = tf.argmax(prediction, axis=1)[0].numpy()
    class_label = class_names[class_index]  # Get the class label
    return class_label


gr.Interface(fn=predict, inputs="image", outputs="text").launch(share=True)
