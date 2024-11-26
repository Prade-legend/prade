import matplotlib.pyplot as p
import tensorflow as tf
from keras.utils import to_categorical as tocat

layers=tf.keras.layers

from keras.datasets import mnist
(xtrain,ytrain),(xtest,ytest)=mnist.load_data()
xtrain=xtrain.astype('float32')/255.0
ytrain=ytrain.astype('float32')/255.0

ytrain=tocat(ytrain,10)
ytest=tocat(ytest,10)

for i in range(1,15):
    p.subplot(5,5,i)
    p.imshow(xtrain[i])
p.show()

nn=tf.keras.Sequential()
nn.add(layers.Flatten())
nn.add(layers.Dense(5,activation='sigmoid'))
nn.add(layers.Dense(10,activation='softmax'))

nn.compile(optimizer='adam',loss='hinge',metrics=['accuracy'])
nn.fit(xtrain,ytrain,epochs=10,batch_size=32,validation_data=(xtest,ytest))

loss,accuracy=nn.evaluate(xtest,ytest)
print("Accuracy and loss",loss,accuracy*100)
