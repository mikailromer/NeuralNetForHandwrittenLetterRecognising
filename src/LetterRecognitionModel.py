
from keras.models import Sequential,load_model
from keras.layers import Dense,Conv2D,MaxPooling2D, Flatten, Dropout
from keras.callbacks import ModelCheckpoint
import keras
from matplotlib import pyplot as plot
from scipy.io import loadmat
from os import path,makedirs
import numpy as np
from random import randint



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
    Model.add(Conv2D(30, (10, 10),padding='same', input_shape=(1, 28, 28), activation='relu'))
    Model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    Model.add(Dropout(0.2))
    Model.add(Conv2D(15, kernel_size=(5, 5),padding='same', activation='relu'))
    Model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    Model.add(Dropout(0.2))
    Model.add(Flatten())
    Model.add(Dense(328,activation='relu'))
    Model.add(Dropout(0.15))
    Model.add(Dense(80, activation='relu'))
    Model.add(Dropout(0.08))
    Model.add(Dense(27,activation='softmax'))
    # Compile model
    Model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return Model

def TrainAndSaveModel(x_train,x_test,y_train,y_test,Epochs,BatchSize,ModelFileName,WeightsFileName):
    FileDir = path.join('', path.dirname(path.abspath(__file__)), 'models')
    if path.isdir(FileDir):
        ModelFilePath=path.join(FileDir,ModelFileName)
        WeightsFilePath=path.join(FileDir,WeightsFileName)
    else:
        makedirs(FileDir)
        ModelFilePath=path.join(FileDir,ModelFileName)
        WeightsFilePath=path.join(FileDir,WeightsFileName)

    Model = NeuralNetInit()
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
    fileDir=path.join('',path.dirname(path.abspath(__file__)),'results')
    if path.isdir(fileDir)==0:
        makedirs(fileDir)
    table=[]
    for i in range(26):
        temp=[]
        for j in range(26):
            temp.append(0)

        table.append(temp)


   # table=np.zeros((26,26),dtype='int')
    for i in range(len(x_test)):
        img = x_test[i]
        test_img = img.reshape((1, 1, 28, 28))
        img_class = Model.predict_classes(test_img)
        prediction = img_class[0]-1
        answer=findAnswer(y_test[i])-1
        table[answer][prediction]=table[answer][prediction]+1


    letterTable=[]
    for i in range(26):
        letterTable.append(chr(i+65))

    table.insert(0,letterTable)
    del letterTable
    for i in range(27):
        if i==0:
            table[i].insert(0, ' ')
        else:
            table[i].insert(0,chr(i+64))
    file=open(path.join(fileDir,'result'),mode='w')
    for row in table:
        for elem in row:
            file.write('%6s' %elem)
        file.write('\n')

    file.close()


    '''
    listOfIndexes=[]
    for i in range(10):
        listOfIndexes.append(randint(0,len(x_test)))

    for i in range(10):
        img=x_test[listOfIndexes[i]]
        answer=findAnswer(y_test[listOfIndexes[i]])
        test_img = img.reshape((1, 1, 28, 28))
        img_class = Model.predict_classes(test_img)
        prediction = convertToLetter(img_class[0])
        answer=convertToLetter(answer)
        print("Prediction: {}\n".format(prediction))
        print("Answer: {}\n".format(answer))
        plot.imshow(img[0], cmap='gray')
        plot.show()
    '''



    # make a predictino
def findAnswer(element):
    answer=None
    for i in range(len(element)):
        if element[i]==1:
            answer=i
            break

    if answer is None:
        raise Exception("The solution doesn't exist.")

    return answer

def convertToLetter(number):
    char= chr(number+96)
    return char

if __name__ == "__main__":
    x_train,x_test,y_train,y_test=loadSamplesAndLabels()
    #PathToTheModel=TrainAndSaveModel(x_train,x_test,y_train,y_test,1000,128,'eminst_mlp_model.h5','emnist.model.best.hdf5')
    ModelEvaluation('/home/michal/Dokumenty/NeuralNetForHandwrittenLetterRecognising/src/models/eminst_mlp_model.h5',x_test,y_test)




