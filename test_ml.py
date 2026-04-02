import tensorflow as tf
import numpy as np
from PIL import Image

model_path = "/Users/athraamosawi/Documents/Hassan/tumor-detection-master/app/brainTumor.keras"
model = tf.keras.models.load_model(model_path)

def predict_img(img_name):
    path = f"/Users/athraamosawi/Documents/Hassan/tumor-detection-master/sample/{img_name}"
    img = Image.open(path).convert('RGB').resize((128, 128))
    img_in = np.array(img).astype("float32") / 255.0
    img_in = np.expand_dims(img_in, axis=0)
    pred = model.predict(img_in, verbose=0)[0][0]
    print(f"{img_name} prediction: {pred}")

predict_img('mri5.jpg')
predict_img('mri1.jpg')

