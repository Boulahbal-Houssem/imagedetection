from keras import backend
assert len(backend.tensorflow_backend._get_available_gpus()) > 0
from model import Model
from dataset.dataset_loader import  Dataset_loader
from preprocessing.preprocessing import Image_processor

import os

if __name__ == "__main__":
	from tensorflow.python.client import device_lib
	print(device_lib.list_local_devices())
	resizer = Image_processor()
	data_loader = Dataset_loader([resizer])
	dataset_path  =os.getcwd() + "/data/train" 
	data, label = data_loader.load(dataset_path,500)
	print("******* Data loaded ********")
	model = Model(data,label,test_size=0.25,batch_size =1)
	model.create_model()
	model.fit()
	model.evaluate()
