#assert len(backend.tensorflow_backend._get_available_gpus()) > 0
from model import Model
from data_generator import DataGenerator, create_dataframe , split_dataframe
import os

if __name__ == "__main__":
    data_path = os.getcwd() + "/data/test"
    data = create_dataframe(data_path)
    trainX, valdX , testX , trainY ,valdY,testY = split_dataframe(data)    
    train_generator = DataGenerator(trainX,trainY,width=222,height=222,channel=3,batch_size=64)
    validation_generator = DataGenerator(trainX,trainY,width=222,height=222,channel=3,batch_size=32)
    test_generator = DataGenerator(trainX,trainY,width=222,height=222,channel=3,batch_size=32)
    model = Model(train_generator,validation_generator,test_generator)
    model.create_model()
    model.summery()
    model.fit()
    model.evaluate()
    
