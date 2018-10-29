from keras import models,layers,optimizers
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt


(train_images,train_labels),(test_images,test_labels)=mnist.load_data()
train_images=train_images.reshape((60000,28,28,1))
train_images=train_images.astype('float32')/255


test_images=test_images.reshape((10000,28,28,1))
test_images=test_images.astype('float32')/255

train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)


model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))


model.compile(optimizer=optimizers.RMSprop(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(train_images,train_labels,epochs=50,batch_size=64,validation_split=0.2)

history_dict=history.history



train_loss=history_dict['loss']
val_loss=history_dict['val_loss']
epochs=range(1,51)
plt.plot(epochs,train_loss,'bo',label='Training Loss')
plt.plot(epochs,val_loss,'b',label='Validation Loss')
plt.title('Loss(alpha=0.001,batch size=64')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
train_acc = history_dict['acc']
val_acc = history_dict['val_acc']
plt.plot(epochs, train_acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()



loss,acc=model.evaluate(test_images,test_labels)

