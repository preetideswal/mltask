#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.layers import Convolution2D


# In[2]:


from keras.layers import MaxPooling2D


# In[3]:


from keras.layers import Flatten


# In[4]:


from keras.layers import Dense


# In[5]:


from keras.models import Sequential


# In[6]:


model = Sequential()


# In[7]:


model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                   input_shape=(64, 64, 3)
                       ))


# In[8]:


model.add(MaxPooling2D(pool_size=(2, 2)))


# In[9]:


model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                       ))


# In[10]:


model.add(MaxPooling2D(pool_size=(2, 2)))


# In[11]:


model.add(Flatten())


# In[12]:


model.add(Dense(units=128, activation='relu'))


# In[30]:


model.add(Dense(units=17, activation='softmax'))


# In[31]:


model.summary()


# In[32]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[33]:


from keras_preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping

# In[47]:


train_data_dir = 'root/mlproject/mltask1/17_flowers/train/'
validation_data_dir = 'root/mlproject/mltask1/17_flowers/validation/'

# Let's use some data augmentaiton 
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=45,
      width_shift_range=0.3,
      height_shift_range=0.3,
      horizontal_flip=True,
      fill_mode='nearest')
 
validation_datagen = ImageDataGenerator(rescale=1./255)
 
# set our batch size (typically on most mid tier systems we'll use 16-32)
batch_size = 32
 
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='categorical')
 
validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(64,64),
        batch_size=batch_size,
        class_mode='categorical')
checkpoint = ModelCheckpoint("root/mlproject/mltask1/flowersvgg.h5",
                             monitor="val_loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)

earlystop = EarlyStopping(monitor = 'val_loss', 
                          min_delta = 0, 
                          patience = 3,
                          verbose = 1,
                          restore_best_weights = True)

# we put our call backs into a callback list
callbacks = [earlystop, checkpoint]
model.compile(loss = 'categorical_crossentropy',
              optimizer = RMSprop(lr = 0.001),
              metrics = ['accuracy'])

# Enter the number of training and validation samples here
nb_train_samples = 1192
nb_validation_samples = 170

# We only train 5 EPOCHS 
epochs = 5
batch_size = 16


history = model.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    callbacks = callbacks,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size)


# In[ ]:


from keras.models import load_model

classifier = load_model('root/mlproject/mltask1/flowers_vgg.h5')


# In[ ]:


import os
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

flowers_dict = {"[0]": "bluebell ", 
                "[1]": "buttercup",
                "[2]": "colts_foot",
                "[3]": "cowslips",
                "[4]": "crocus",
                "[5]": "daffodil",
                "[6]": "daisy",
                "[7]": "dandelion",
                "[8]": "fritillay",
                "[9]": "iris",
                "[10]": "lily_valley ",
                "[11]": "pansy ",
                "[12]": "snowdrop",
                "[13]": "sunflower",
                "[14]": "tigerlily",
                "[15]": "tulip",
                "[16]":"windflower"}

flowers_dict_n = {"bluebell":"bluebell ", 
                      "buttercup": "buttercup",
                      "colts_foot": "colts_foot",
                      "cowslips": "cowslips",
                      "crocus": "crocus",
                      "daffodil": "daffodil",
                      "daisy": "daisy",
                      "dandelion": "dandelion",
                      "fritillary": "fritillary",
                      "iris": "iris",
                      "lily_valley": "lily_valley",
                      "pansy": "pansy",
                      "snowdrop": "snowdrop",
                      "sunflower": "sunflower",
                      "tigerlily": "tigerlily",
                      "tulip": "tulip",
                      "windflower":"windflower"}

def draw_test(name, pred, im):
    flower = flowers_dict[str(pred)]
    BLACK = [0,0,0]
    expanded_image = cv2.copyMakeBorder(im, 80, 0, 0, 100 ,cv2.BORDER_CONSTANT,value=BLACK)
    cv2.putText(expanded_image, flower, (20, 60) , cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2)
    cv2.imshow(name, expanded_image)

def getRandomImage(path):
    """function loads a random images from a random folder in our test path """
    folders = list(filter(lambda x: os.path.isdir(os.path.join(path, x)), os.listdir(path)))
    random_directory = np.random.randint(0,len(folders))
    path_class = folders[random_directory]
    print("Class - " + flowers_dict_n[str(path_class)])
    file_path = path + path_class
    file_names = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    random_file_index = np.random.randint(0,len(file_names))
    image_name = file_names[random_file_index]
    return cv2.imread(file_path+"/"+image_name)    

for i in range(0,10):
    input_im = getRandomImage("root/mlproject/mltask1/17_flowers/validation/")
    input_original = input_im.copy()
    input_original = cv2.resize(input_original, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
    
    input_im = cv2.resize(input_im, (224, 224), interpolation = cv2.INTER_LINEAR)
    input_im = input_im / 255.
    input_im = input_im.reshape(1,224,224,3) 
    
    # Get Prediction
    res = np.argmax(classifier.predict(input_im, 1, verbose = 0), axis=1)
    
    # Show image with predicted class
    draw_test("Prediction", res, input_original) 
    cv2.waitKey(0)

cv2.destroyAllWindows()







