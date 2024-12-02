from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image 
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os
#image data generator is the package to lable the images & it will automatically lable all the images

import warnings
warnings.filterwarnings

img = image.load_img(r"C:\Users\gadel\VS Code projects\Emotion Detector\CNN\Testing\Screenshot (852).png")

plt.imshow(img)

i1 = cv2.imread(r"C:\Users\gadel\VS Code projects\Emotion Detector\CNN\Testing\Screenshot (852).png")
i1
# 3 dimension metrics are created for the image
# the value ranges from 0-255

i1.shape
# shape of your image height, weight, rgb

train = ImageDataGenerator(rescale = 1/255)
validataion = ImageDataGenerator(rescale = 1/255)
# to scale all the images i need to divide with 255
# we need to resize the image using 200, 200 pixel

train_dataset = train.flow_from_directory(r"C:\Users\gadel\VS Code projects\Emotion Detector\CNN\Training",
                                         target_size = (200,200),
                                         batch_size = 3,
                                         class_mode = 'binary')
validataion_dataset = validataion.flow_from_directory(r"C:\Users\gadel\VS Code projects\Emotion Detector\CNN\Validation",
                                          target_size = (200,200),
                                          batch_size = 3,
                                          class_mode = 'binary')

train_dataset.class_indices

train_dataset.classes

# now we are applying maxpooling 

model = tf.keras.models.Sequential([ tf.keras.layers.Conv2D(16,(3,3),activation = 'relu',input_shape = (200,200,3)),
                                    tf.keras.layers.MaxPool2D(2,2), #3 filtr we applied hear
                                    #
                                    tf.keras.layers.Conv2D(32,(3,3),activation = 'relu'),
                                    tf.keras.layers.MaxPool2D(2,2),    
                                    #                       
                                    tf.keras.layers.Conv2D(64,(3,3),activation = 'relu'),
                                    tf.keras.layers.MaxPool2D(2,2), 
                                    ##
                                    tf.keras.layers.Flatten(),
                                    ##
                                    tf.keras.layers.Dense(512, activation = 'relu'),
                                    #
                                    tf.keras.layers.Dense(1,activation= 'sigmoid')
                                    ]
                                    )

model.compile(loss='binary_crossentropy',
              optimizer = tf.keras.optimizers.RMSprop(lr = 0.001),
              metrics = ['accuracy']
              )

model_fit = model.fit(train_dataset,
                     steps_per_epoch = 3,
                     epochs = 10,
                     validation_data = validataion_dataset)

dir_path = r"C:\Users\gadel\VS Code projects\Emotion Detector\CNN\Testing"
for i in os.listdir(dir_path ):
    print(i)
    #img = image.load_img(dir_path+ '//'+i, target_size = (200,200))
   # plt.imshow(img)
   # plt.show()
   
dir_path = r"C:\Users\gadel\VS Code projects\Emotion Detector\CNN\Testing"
for i in os.listdir(dir_path ):
    img = image.load_img(dir_path+ '//'+i, target_size = (200,200))
    plt.imshow(img)
    plt.show()
    
dir_path = r"C:\Users\gadel\VS Code projects\Emotion Detector\CNN\Testing"
for i in os.listdir(dir_path ):
    img = image.load_img(dir_path+ '//'+i, target_size = (200,200))
    plt.imshow(img)
    plt.show()
        
    x= image.img_to_array(img)
    x=np.expand_dims(x,axis = 0)
    images = np.vstack([x])
    
    val = model.predict(images)
    if val == 0:
        print( ' i am not happy')
    else:
        print('i am happy')