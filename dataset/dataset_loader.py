import os
import numpy as np
from skimage import io

image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
def list_images(basePath, contains=None):
    # return the set of files that are valid
    return list_files(basePath, validExts=image_types, contains=contains)


def list_files(basePath, validExts=None, contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if validExts is None or ext.endswith(validExts):
                # construct the path to the image and yield it
                imagePath = os.path.join(rootDir, filename)
                yield imagePath


class Dataset_loader():
    data_ =[]
    label_=[]
    def __init__(self,preprocessors=None):
        self.preprocessors= preprocessors

    def load(self,p,verbose =-1):
        paths =  list_images(p)
        for (i,image_path) in enumerate(paths):
            image = io.imread(image_path)
            label = image_path.split(os.path.sep)[-1].split(".")[0]
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.process(image)
            Dataset_loader.data_.append(image)
            Dataset_loader.label_.append(label)
            if(verbose >0 and (i%verbose) == 0):
                print("[INFO] processed " +str(i))

        return (np.array(Dataset_loader.data_),np.array(Dataset_loader.label_))
