
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
import tensorflow as tf

model = load_model('perfect_models/Stroke.h5')
# class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

class_names =['stroke_no', 'stroke_yes']
image_path = 'test_image/x-ray.jpeg'
# image_path = 'test_image/rose2.jpeg'
img = load_img(
    image_path, target_size=(180, 180)
)
img_array = img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
