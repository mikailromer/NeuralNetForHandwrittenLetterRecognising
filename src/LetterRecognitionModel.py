
from keras.models import Sequential,load_model
from keras.layers import Dense,Conv2D,MaxPooling2D, Flatten, Dropout
from keras.callbacks import ModelCheckpoint
import keras
from matplotlib import pyplot as plot
from scipy.io import loadmat
from os import path,makedirs
import numpy as np



def loadSamplesAndLabels():
    '''
        Function performs the following steps:
            1.Loading input samples of images and appropiate labels for them, from emnist-letters.mat file.
            2.Normalization of input samples and their output labels
            3.Reshaping data
    :return:List of input saples
    '''
    PathToImageSamples = path.join('', path.dirname(path.abspath(__file__)), 'samples', 'emnist-letters.mat')
    emnist=loadmat(PathToImageSamples)
    #Loading train labels
    x_train = emnist["dataset"][0][0][0][0][0][0]
    x_train = x_train.astype(np.float32)
    y_train = emnist["dataset"][0][0][0][0][0][1]
    #Loading test labels
    x_test = emnist["dataset"][0][0][1][0][0][0]
    x_test = x_test.astype(np.float32)
    y_test = emnist["dataset"][0][0][1][0][0][1]
    #Normalization of input samples
    x_train/=255
    x_test/=255
    #Normalization of output labels
    y_train=keras.utils.to_categorical(y_train,num_classes=27)
    y_test = keras.utils.to_categorical(y_test,num_classes=27)
    #Reshaping train labels
    x_train=x_train.reshape(x_train.shape[0], 1, 28, 28, order="A")
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28, order="A")
    return x_train,x_test,y_train,y_test

def NeuralNetInit():
    Model=Sequential()
    Model.add(Conv2D(30, (6, 6),padding='same', input_shape=(1, 28, 28), activation='relu'))
    Model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    Model.add(Dropout(0.2))
    Model.add(Conv2D(15, kernel_size=(3, 3),padding='same', activation='relu'))
    Model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    Model.add(Dropout(0.2))
    Model.add(Flatten())
    Model.add(Dense(256,activation='relu'))
    Model.add(Dropout(0.15))
    Model.add(Dense(56, activation='relu'))
    Model.add(Dropout(0.08))
    Model.add(Dense(27,activation='softmax'))
    # Compile model
    Model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return Model

def TrainAndSaveModel(x_train,x_test,y_train,y_test,Epochs,BatchSize,ModelFileName,WeightsFileName):
    Model=NeuralNetInit()
    FileDir = path.join('', path.dirname(path.abspath(__file__)), 'models')
    if path.isdir(FileDir):
        ModelFilePath=path.join(FileDir,ModelFileName)
        WeightsFilePath=path.join(FileDir,WeightsFileName)
    else:
        makedirs(FileDir)
        ModelFilePath=path.join(FileDir,ModelFileName)
        WeightsFilePath=path.join(FileDir,WeightsFileName)

    checkpointer=ModelCheckpoint(WeightsFilePath,verbose=1,save_best_only=True)
    Model.fit(x_train,y_train,batch_size=BatchSize, epochs=Epochs, validation_split=0.2,callbacks=[checkpointer],verbose=1,shuffle=True)
    Model.load_weights(path.join('',FileDir,WeightsFilePath))
    Model.save(ModelFilePath)
    return ModelFilePath

def ModelEvaluation(ModelFileName, x_test,y_test):
    Model=load_model(ModelFileName)
    result=Model.evaluate(x_test,y_test,verbose=0)
    accuracy=100*result[1]
    print('Test accuracy: %.4f%%' % accuracy)

x_train,x_test,y_train,y_test=loadSamplesAndLabels()
PathToTheModel=TrainAndSaveModel(x_train,x_test,y_train,y_test,1000,128,'eminst_mlp_model.h5','emnist.model.best.hdf5')
ModelEvaluation(PathToTheModel,x_test,y_test)



#img = x_train[1]
#answer = y_train[1]
# visualize image
#plot.imshow(img[0], cmap='gray')

