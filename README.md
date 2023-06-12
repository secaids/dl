# Datasets
## MNIST
The MNIST dataset is a collection of handwritten digits. The dataset has a collection of 70,000 handwrittend digits of size 28 X 28.

The dataset is divided into two main parts: the training set and the test set. The training set contains 60,000 images, while the test set consists of 10,000 images. This split allows researchers to train their models on a large amount of data and then evaluate their performance on unseen examples.

Each image in the MNIST dataset is labeled with the corresponding digit it represents, ranging from 0 to 9.The dataset has balanced class distributions, meaning that each digit appears roughly the same number of times in the dataset.

The MNIST dataset has been a popular choice for beginners in machine learning due to its simplicity and well-defined problem.
## CIFAR - 10
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

Here are the classes in the dataset: Airplane, Automobile, Nird, Car, Deer, Dog, Frog, Horse, Ship, Truck
## Malaria
The dataset contains 2 folders: Infected, Uninfected. And a total of 27,558 images.

Malaria dataset of 27,558 cell images with an equal number of parasitized and uninfected cells. 

A level-set based algorithm was applied to detect and segment the red blood cells. The images were collected and annotated by medical professionals.
## NER
We aim to develop an LSTM-based neural network model using Bidirectional Recurrent Neural Networks for recognizing the named entities in the text.

The dataset used has a number of sentences, and each words have their tags.

We have to vectorize these words using Embedding techniques to train our model.

Bidirectional Recurrent Neural Networks connect two hidden layers of opposite directions to the same output.
##  Stocks
Time Series data is a series of data points indexed in time order. Time series data is everywhere, so manipulating them is important for any data analyst or data scientist.

In this notebook, we have stock data from the stock market, particularly some technology stocks (Apple, Amazon, Google, and Microsoft).

For this we are provided with a dataset which contains features like
Date, Opening Price, Highest Price, Lowest Price, Closing Price, Adjusted Closing Price, Volume
Based on the given features, develop a RNN model to predict, the price of stocks in future

## NN - Regression Model
```py
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import RootMeanSquaredError

ai = Seq([
    Den(8,activation = 'relu',input_shape=[1]),
    Den(15,activation = 'relu'),
    Den(1),
])

ai.compile(optimizer = 'rmsprop',loss = 'mse')

ai.fit(x_train,y_train,epochs=2000)

err = rmse()
preds = ai.predict(x_test)
err(y_test,preds)
```
## NN - Classification Model
```py
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv("./customers.csv")

ai = Sequential([Dense(50,input_shape = [8]),
                 Dense(40,activation="relu"),
                 Dense(30,activation="relu"),
                 Dense(20,activation="relu"),
                 Dense(4,activation="softmax")])

ai.compile(optimizer='adam',
           loss='categorical_crossentropy',
           metrics=['accuracy'])

early_stop = EarlyStopping(
    monitor='val_loss',
    mode='max', 
    verbose=1, 
    patience=20)
    
ai.fit( x = x_train, y = y_train,
        epochs=500, batch_size=256,
        validation_data=(x_test,y_test),
        callbacks = [early_stop]
        )
        
x_pred = np.argmax(ai.predict(x_test), axis=1)
y_truevalue = np.argmax(y_test,axis=1)
conf(y_truevalue,x_pred)
print(report(y_truevalue,x_pred))
```
## Digit Classification
```py
from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils

(x_train, y_train), (x_test, y_test) = mnist.load_data()

model = keras.Sequential()
input = keras.Input(shape=(28,28,1))
model.add(input)

model.add(layers.Conv2D(filters=32,kernel_size=(5,5),
			strides=(1,1),padding='valid',activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Conv2D(filters=64,kernel_size=(5,5),
			strides=(1,1),padding='same',activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(20,activation='relu'))
model.add(layers.Dense(15,activation='relu'))
model.add(layers.Dense(5,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train_scaled ,y_train_onehot, epochs=5,batch_size=64, 
          validation_data=(X_test_scaled,y_test_onehot))

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
print(confusion_matrix(y_test,x_test_predictions))
print(classification_report(y_test,x_test_predictions))
```
## Transfer Learning
```py
from keras import Sequential
from keras.layers import Flatten,Dense,BatchNormalization,Activation,Dropout
from tensorflow.keras import utils
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

(x_train,y_train),(x_test,y_test)=cifar10.load_data()

base_model = VGG19(include_top=False, weights = "imagenet",
                   input_shape = (32,32,3))
                   
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(250,activation=("relu")))
model.add(Dropout(0.2))
model.add(Dense(100,activation=("relu")))
model.add(Dropout(0.35))
model.add(Dense(10,activation=("softmax")))
model.summary()

model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',patience=3,
                                            verbose=1,factor=0.5,min_lr=0.00001)

model.fit(x_train, y_train, batch_size=500, epochs=10, validation_data=(x_test, y_test), 
          callbacks=[learning_rate_reduction])

x_test_predictions = np.argmax(model.predict(x_test), axis=1)
print(confusion_matrix(y_test,x_test_predictions))
print(classification_report(y_test,x_test_predictions))
```
## Stock Price Predicition
```py
from keras import layers
from keras.models import Sequential

dataset_train = pd.read_csv('trainset.csv')

model = Sequential([layers.SimpleRNN(50,input_shape=(60,1)),
                    layers.Dense(1)])

model.compile(optimizer='adam',loss='mse')
model.summary()

model.fit(X_train1,y_train,epochs=20, batch_size=32)

predicted_stock_price_scaled = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price_scaled)
from sklearn.metrics import mean_squared_error as mse
mse(y_test,predicted_stock_price)
```
## NER
```py
from keras import layers
from keras.models import Model
from tensorflow.keras.preprocessing import sequence

data = pd.read_csv("ner_dataset.csv", encoding="latin1")

input_word = layers.Input(shape=(max_len,))

embedding_layer = layers.Embedding(input_dim=num_words,output_dim=50,
                                   input_length=max_len)(input_word)
dropout = layers.SpatialDropout1D(0.1)(embedding_layer)

bid_lstm = layers.Bidirectional(
    layers.LSTM(units=100,return_sequences=True,
                recurrent_dropout=0.1))(dropout)

output = layers.TimeDistributed(
    layers.Dense(num_tags,activation="softmax"))(bid_lstm)

model = Model(input_word, output)  

model.summary()

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(
    x=X_train, y=y_train, validation_data=(X_test,y_test),
    batch_size=50, epochs=3,
)
```
## AutoEncoder
```py
from tensorflow import keras
from tensorflow.keras import layers, utils, models
from tensorflow.keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()

input_img = keras.Input(shape=(28, 28, 1))

x=layers.Conv2D(32,(3,3),activation='relu',padding='same')(input_img)
x=layers.MaxPooling2D((2, 2), padding='same')(x)
x=layers.Conv2D(32,(3,3),activation='relu',padding='same')(x)

encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

x=layers.Conv2D(32,(3,3),activation='relu',padding='same')(encoded)
x=layers.UpSampling2D((2,2))(x)
x=layers.Conv2D(32,(3,3),activation='relu',padding='same')(x)
x=layers.UpSampling2D((2,2))(x)

decoded = layers.Conv2D(1, (3, 3), activation='sigmoid',padding='same')(x)

autoencoder = keras.Model(input_img, decoded)

autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train_noisy, x_train_scaled,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test_scaled))
```
## Malaria
```py
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.compat.v1.keras.backend import set_session

my_data_dir = "./cell_images"
test_path = my_data_dir+'/test/'
train_path = my_data_dir+'/train/'
para_img= imread(train_path+'/parasitized/'+os.listdir(train_path+'/parasitized')[0])

model = models.Sequential()
model.add(keras.Input(shape=(image_shape)))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu',))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu',))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(128))
model.add(layers.Dense(64,ativation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

batch_size = 16

results = model.fit(train_image_gen,epochs=2,validation_data=test_image_gen)

model.evaluate(test_image_gen)
pred_probabilities = model.predict(test_image_gen)
test_image_gen.classes
predictions = pred_probabilities > 0.5
print(classification_report(test_image_gen.classes,predictions))
confusion_matrix(test_image_gen.classes,predictions)
```
