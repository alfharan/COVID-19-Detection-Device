# Created by:
# Name : Alan Fhajoeng Ramadhan
# Country : Indonesia
# email : alfhatech.id@gmail.com
# 21 March 2020

import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

CATEGORIES = ["covid", "normal"]
def prepare(filepath):
    IMG_SIZE = 500
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model("PATH")

image = cv2.imread('1.jpg', cv2.IMREAD_COLOR)
image = cv2.resize(image, (1500,1500))
img_array = cv2.imread('1.jpg', cv2.IMREAD_COLOR)
img_array = cv2.resize(img_array, (1500,1500))

prediction = model.predict([prepare('1.jpg')])
label = str(CATEGORIES[int(prediction[0][0])])
if label == 'covid':
    cv2.putText(img_array, "Suspect : Positive COVID-19", (50,300),  cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,0), 5, cv2.LINE_AA)

else:
    cv2.putText(img_array, "Suspect : Normal", (50,300),  cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,0), 5, cv2.LINE_AA)

plt.imshow(image)
plt.show()
plt.imshow(img_array)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

# Delete this if you run it
Created by:
Name : Alan Fhajoeng Ramadhan
Country : Indonesia
email : alfhatech.id@gmail.com
21 March 2020




