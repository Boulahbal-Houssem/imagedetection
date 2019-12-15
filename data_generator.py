from keras.utils.data_utils import Sequence 
from keras.utils import to_categorical
import os
from sklearn.model_selection import train_test_split
from skimage import io
import numpy as np
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder


def create_dataframe(path):
        files  =  list(os.walk(path))[0][2]
        ret = []
        for file in files:
            ret.append( [path+"/"+file,file.split(".")[0] ] )
        return np.array(ret)
        
def split_dataframe(df):
    trainX,vald_testX,trainY,vald_testY = train_test_split(df[:,0],df[:,1],test_size=0.25)
    valdX,testX,valdY,testY = train_test_split(vald_testX,vald_testY,test_size=0.5)
    return trainX, valdX , testX , trainY ,valdY,testY

class DataGenerator(Sequence):
    def __init__(self, data, labels,width,height,channel, batch_size=32,shuffle=True):
        self.batch_size = batch_size
        self.labels = labels
        self.data = data
        self.shuffle = shuffle
        self.on_epoch_end()
        self.width = width
        self.height = height
        self.channel = channel
         
    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.data[k] for k in indexes]
        X = self.__data_generation(list_IDs_temp)
        
        y = [self.labels[k] for k in indexes]
        lb = LabelEncoder()
        y = lb.fit_transform(y)
        return (X, to_categorical(y))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, file):
        shape =  (self.batch_size, self.width,self.height,self.channel)
        X = np.empty(shape)
        for i, img in enumerate(file):
            image = io.imread(img)
            X[i] = resize(image,(self.width,self.height),anti_aliasing=True)
        return X 

    @property
    def shape(self):
        return (self.batch_size, self.width,self.height,self.channel)


