import tensorflow as tf
import numpy as np
from PIL import Image

model_save_path = '/Users/huangrenming/Desktop/data/checkpoint/mnist.ckpt'

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.load_weights(model_save_path)
for i in range(10):
    image_path = input('input image path:')
    img = Image.open(image_path)
    img_arr = np.array(img.convert('L'))
    # img_arr = 255- img_arr

    img_arr = img_arr / 255.0
    x_predict = img_arr[tf.newaxis, ...]
    result = model.predict(x_predict)
    pred = tf.argmax(result, axis=1)
    
    tf.print(pred)
    print('\n')


