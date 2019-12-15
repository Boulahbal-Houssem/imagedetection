
from keras import Sequential
from keras.utils.np_utils import to_categorical
from keras.layers import Conv2D, MaxPool2D, Input, Dense, BatchNormalization,Activation,Flatten
from keras.optimizers import adam
from keras.losses import categorical_crossentropy
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from os import getcwd

log_dir = getcwd() +"/models_log"
checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
	monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

class Model():
	def __init__(self, train_generator,validation_generator,test_generator,epochs=10):
		self.train_generator = train_generator
		self.validation_generator = validation_generator
		self.test_generator = test_generator
		self.batch_size = len(train_generator)
		self.epochs= epochs

	def create_model(self):
		input_shape_ = self.train_generator.shape[1:]
		self.model = Sequential()
		self.model.add(Conv2D(128, (3,3),strides=(1,1), padding='same', input_shape=input_shape_ ))
		self.model.add(Activation('relu'))
		self.model.add(Conv2D(64, (3,3),strides=(2,2), padding='same', input_shape=[32,32,1] ))
		self.model.add(Activation('relu'))
		self.model.add(Conv2D(32, (3,3),strides=(2,2), padding='same', input_shape=[32,32,1] ))
		self.model.add(Activation('relu'))
		self.model.add(Flatten())
		self.model.add(Dense(64))
		self.model.add(Activation('relu'))
		self.model.add(Dense(32))
		self.model.add(Activation('relu'))
		self.model.add(Dense(2))
		self.model.add(Activation('softmax'))
		self.summery()
		opt = adam(0.01)
		self.model.compile(optimizer=opt,loss=categorical_crossentropy,metrics=['accuracy'])

	def summery(self):
		self.model.summary()

	def fit(self):

		self.model.fit_generator(generator=self.train_generator,epochs=self.epochs,
                    			validation_data=self.validation_generator,
                    			use_multiprocessing=True,
                    			workers=4,callbacks = [checkpoint,reduce_lr,early_stopping])
		self.model.save("save")

	def evaluate(self):
		score, acc = self.model.evaluate_generator(generator=self.test_generator,
                                              max_queue_size=10, workers=1, use_multiprocessing=False,
                                              verbose=0)
		print('Test score:', score)
		print('Test accuracy:', acc)


		
		
