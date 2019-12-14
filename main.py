from model import Model
from dataset.dataset_loader import  Dataset_loader
from preprocessing.preprocessing import Image_processor

import os

if __name__ == "__main__":
	from tensorflow.python.client import device_lib
	print(device_lib.list_local_devices())
	resizer = Image_processor()
	data_loader = Dataset_loader([resizer])
	dataset_path  =os.getcwd() + "/data/test" 
	data, label = data_loader.load(dataset_path)
	model = Model(data,label,test_size=0.25,batch_size =1)
	model.create_model()
	model.fit()
	model.evaluate()
