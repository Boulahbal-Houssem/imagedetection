
from keras import Sequential
from keras.utils.np_utils import to_categorical
from keras.layers import Conv2D, MaxPool2D, Input, Dense, BatchNormalization,Activation,Flatten
from keras.optimizers import adam
from keras.losses import categorical_crossentropy
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def normalization(trainX,testX,trainY,testY):
	trainX = trainX.astype('float32')
	testX = testX.astype('float32')
	trainX /= 255
	testX /= 255
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX,testX,trainY,testY

class Model():
	def __init__(self, data, label, test_size = 0.25, batch_size = 32, epochs=10):
		self.data = data
		self.label = label
		self.test_size =test_size
		self.batch_size = batch_size
		self.epochs= epochs
		self.ecnoding()

	def create_model(self):
		input_shape_ = self.trainX.shape[1:]
		self.model = Sequential()
		self.model.add(Conv2D(64, (3,3),strides=(1,1), padding='same', input_shape=input_shape_ ))
		self.model.add(Activation('relu'))
		self.model.add(Conv2D(32, (3,3),strides=(1,1), padding='same', input_shape=[32,32,1] ))
		self.model.add(Activation('relu'))
		self.model.add(Conv2D(16, (3,3),strides=(2,2), padding='same', input_shape=[32,32,1] ))
		self.model.add(Activation('relu'))
		self.model.add(Flatten())
		self.model.add(Dense(64))
		self.model.add(Activation('relu'))
		self.model.add(Dense(32))
		self.model.add(Activation('relu'))
		self.model.add(Dense(2))
		self.model.add(Activation('softmax'))
		self.summery()
		opt = adam(0.001)
		self.model.compile(optimizer=opt,loss=categorical_crossentropy,metrics=['accuracy'])

	def summery(self):
		self.model.summary()

	def fit(self):
		self.trainX,self.testX,self.trainY,self.testY = normalization(self.trainX,self.testX,self.trainY,self.testY)
		print(self.trainY)
		self.model.fit(self.trainX, self.trainY,self.batch_size,self.epochs,validation_data=(self.testX, self.testY),shuffle=True)
		self.model.save("save")

	def ecnoding(self):
		lb = LabelEncoder()
		self.label = lb.fit_transform(self.label)
		self.trainX,self.testX,self.trainY,self.testY= train_test_split(self.data,self.label,test_size=self.test_size,random_state=42)

	def evaluate(self):
		score, acc = self.model.evaluate(self.testX, self.testY, self.batch_size)
		print('Test score:', score)
		print('Test accuracy:', acc)


		
		
