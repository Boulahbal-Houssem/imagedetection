#!/bin/bash 

# install depedecies 
pip3 install kaggle
pip3 install keras
pip3 install tensorflow
pip3 install scikit-image

# setting up kaggle colab

sudo mkdir /root/.kaggle/
#sudo ln -s ~/.local/bin/kaggle /usr/bin/kaggle
#sudo cp ./kaggle.json /home/houssem/.kaggle/kaggle.json
sudo cp ./kaggle.json /root/.kaggle/kaggle.json
chmod 600 /root/.kaggle/kaggle.json

# Download dataset 
mkdir -p data
cd data
kaggle competitions download -c dogs-vs-cats
unzip dogs-vs-cats
rm dogs-vs-cats.zip sampleSubmission.csv test1.zip
unzip train





