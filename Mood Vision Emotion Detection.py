from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os

import warnings
warnings.filterwarnings

img = image.load_img(r"C:\Users\chitt\OneDrive\Desktop\CNN\Training\happy\download (4).jpg")
plt.imshow(img)
i1 = cv2.imread(r"C:\Users\chitt\OneDrive\Desktop\CNN\Training\happy\download (4).jpg")
i1

i1.shape


train = ImageDataGenerator(rescale =1/200)
validation = ImageDataGenerator(rescale =1/200)

train_dataset = train.flow_from_directory(r"C:\Users\chitt\OneDrive\Desktop\CNN\Training",
                                          target_size = (200,200),
                                          batch_size =32,
                                          class_mode ='binary')
validation_dataset = validation.flow_from_directory(r"C:\Users\chitt\OneDrive\Desktop\CNN\Velidation",
                                          target_size = (200,200),
                                          batch_size =32,
                                          class_mode ='binary')
train_dataset.class_indices
train_dataset.classes
                                   
model = tf.keras.models.Sequential([ tf.keras.layers.Conv2D(16,(3,3),activation = 'relu',input_shape = (None,200,200,3)),
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
              optimizer = tf.keras.optimizers.RMSprop(learning_rate= 0.001),
              metrics = ['accuracy']
              )

dir_path = r"C:\Users\chitt\OneDrive\Desktop\CNN\Testing"
for i in os.listdir(dir_path):
    print(i)
    
model_fit = model.fit(train_dataset,
                      steps_per_epoch=3,
                      epochs=10,
                      validation_data=validation_dataset)

dir_path= r"C:\Users\chitt\OneDrive\Desktop\CNN\Testing"
for i in os.listdir(dir_path):
    print(i)
    
    
dir_path = r"C:\Users\chitt\OneDrive\Desktop\CNN\Testing"
for i in os.listdir(dir_path):
    img = image.load_img(dir_path+ '//'+i, target_size =(200,200))
    plt.imshow(img)
    plt.show()
    

dir_path = r"C:\Users\chitt\OneDrive\Desktop\CNN\Testing"
for i in os.listdir(dir_path):
    img = image.load_img(dir_path+ '//'+i, target_size =(200,200))
    plt.imshow(img)
    plt.show()
    x= image.img_to_array(img)
    x=np.expand_dims(x,axis =0)
    images=np.vstack([x])
    val=model.predict(images)
    if val==0:
        print('i am happy')
    else:
        print('i am not happy')
        
# Save the model in H5 format
model.save("moodvision_model.h5")

import os

# Get the current working directory
current_directory = os.getcwd()
print(f"Current Working Directory: {current_directory}")

# Define a function to preprocess the uploaded image
def preprocess_image(uploaded_image):
    img = Image.open(uploaded_image).convert('RGB')  # Ensure the image is in RGB format (3 channels)
    img = img.resize((200, 200))  # Resize to match model input shape
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    
    # Ensure the image has 3 channels for RGB compatibility
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (shape: (1, 200, 200, 3))
    
    return img_array

        

