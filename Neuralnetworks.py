import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as py
import sklearn

from sklearn import model_selection
import pandas
import numpy
import pickle
import h5py

data=keras.datasets.fashion_mnist
x,y=data.load_data()
print(data)
https://www.youtube.com/watch?v=x6f5JOPhci0   (X is weight y is cost)
df=pandas.DataFrame(data)
pandas.set_option("display.max_columns",500)
print(df.columns)
print(df)
X=df[]
y=

#trainimages, trainlabels , testimages, testlabels = sklearn.model_selection.train_test_split(X,y,test_size=0.1)
(trainimages, trainlabels) , (testimages, testlabels) =  data.load_data()
print(testlabels[5:20])
trainimages=trainimages/255
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
model=keras.Sequential([keras.layers.Flatten(input_shape=(28,28)), keras.layers.Dense(128,activation='relu'), keras.layers.Dense(10,activation='softmax')])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(trainimages,trainlabels,epochs=1)

##model.save('Neuralmodel.h5')
model=keras.models.load_model('Neuralmodel.h5')
test_loss, test_acc  = model.evaluate(testimages,testlabels)
print("Test loss: ",test_loss,"Test Accuracy: ",test_acc)

prediction=model.predict(numpy.array(testimages))
for i in range(5):
    py.imshow(testimages[i])
    py.xlabel(class_names[testlabels[i]])
    py.title(class_names[numpy.argmax(prediction[i])])
 
print("Predicted Value: ",class_names[numpy.argmax(prediction[i])],"Orignal Value: ",testlabels[i])
