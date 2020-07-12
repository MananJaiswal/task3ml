from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

IMAGE_SIZE = [224,224]

from google.colab import drive
drive.mount('/content/drive')

train_path = '/content/drive/My Drive/data/train'
valid_path = '/content/drive/My Drive/data/val'

vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

for layer in vgg.layers:
    layer.trainable = False

folders = glob('/content/drive/My Drive/data/train/*')

vgg.layers

x = Flatten()(vgg.output)
top_model = Dense(len(folders), activation='softmax')(x)

model = Model(inputs=vgg.input, outputs=top_model)


model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


from keras_preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/content/drive/My Drive/data/train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('/content/drive/My Drive/data/val',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# In[14]:


r=model.fit_generator(training_set,
                         samples_per_epoch = 64,
                         nb_epoch = 5,
                         validation_data = test_set,
                         nb_val_samples = 32)

r=test_set.class_indices
s=training_set.class_indices
print(r)
print(s)

model.save("facerecog_vgg.h5")

from keras.models import load_model
classifier = load_model('facerecog_vgg.h5')

import os
from google.colab.patches import cv2_imshow
import numpy as np
from os import listdir
from os.path import isfile, join

face_recogs_dict = {"[0]": "ben_afflek",
                   "[1]": "elton_john"}

face_recogs_dict_n = {"ben_afflek": "ben_afflek", 
                      "elton_john": "elton_john"}

def draw_test(name, pred, im):
    person = face_recogs_dict[str(pred)]
    BLACK = [0,0,0]
    expanded_image = cv2.copyMakeBorder(im, 80, 0, 0, 100 ,cv2.BORDER_CONSTANT,value=BLACK)
    cv2.putText(expanded_image, person, (20, 60) , cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2)
    
def getRandomImage(path):
    """function loads a random images from a random folder in our test path """
    folders = list(filter(lambda x: os.path.isdir(os.path.join(path, x)), os.listdir(path)))
    random_directory = np.random.randint(0,len(folders))
    path_class = folders[random_directory]
    print("Class - " + face_recogs_dict_n[str(path_class)])
    file_path = path + path_class
    file_names = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    random_file_index = np.random.randint(0,len(file_names))
    image_name = file_names[random_file_index]
    return cv2.imread(file_path+"/"+image_name)

for i in range(0,2):    
    input_im = getRandomImage("/content/drive/My Drive/data/val/")
    input_original = input_im.copy()
    input_original = cv2.resize(input_original, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
    
    input_im = cv2.resize(input_im, (224, 224), interpolation = cv2.INTER_LINEAR)
    input_im = input_im / 255.
    input_im = input_im.reshape(1,224,224,3) 
    
    # Get Prediction
    res = np.argmax(classifier.predict(input_im, 1, verbose = 0), axis=1)
    
    # Show image with predicted class
    draw_test("Prediction", res, input_original) 
    cv2.waitKey(1000)

cv2.destroyAllWindows()
