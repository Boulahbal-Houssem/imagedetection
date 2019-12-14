from skimage import data, color
from skimage.transform import resize
class Image_processor():
	def __init__(self,w=100,h=100,c=3):
		self.width = w
		self.height =h
		self.channels = c
	def process(self,image):
		return resize(image,(self.width,self.height),anti_aliasing=True)
