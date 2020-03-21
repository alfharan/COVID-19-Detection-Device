# Created by:
# Name : Alan Fhajoeng Ramadhan
# From : Indonesia
# email : alfhatech.id@gmail.com
# 21 March 2020

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 as cv
import random
import pickle

DATADIR = r'PATH'  # your Training Data folder path
CATEGORIES = ["covid", "normal"]

for category in CATEGORIES:
    path = os.path.join(DATADIR, category) # path to CATEGORIES
    for img in os.listdir(path):
        img_array = cv.imread(os.path.join(path,img), cv.IMREAD_COLOR)
        plt.imshow(img_array)
        plt.show()
        break
    break 
print(img_array.shape)
IMG_SIZE = 500

new_array = cv.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array,cmap ='gray')
plt.show()

training_data =[]

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) # path to CATEGORIES
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv.imread(os.path.join(path,img), cv.IMREAD_GRAYSCALE)
                new_array = cv.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
            
create_training_data()

print(len(training_data))

random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample[1])
X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
pickle_out =open("X.pickle", "wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out =open("y.pickle", "wb")
pickle.dump(y,pickle_out)
pickle_out.close()

Created by:
Name : Alan Fhajoeng Ramadhan
From : Indonesia
email : alfhatech.id@gmail.com
21 March 2020
