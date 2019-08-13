# Food-Recognition-using-DenseNet-and-MobileNetV2
## 1. About
This repo contains the source code I used to conduct the various experiments in my thesis. Most of the code focuses on training [DenseNet](https://arxiv.org/pdf/1608.06993.pdf) and [MobileNetV2](https://arxiv.org/pdf/1801.04381.pdf) CNNs for food recognition using the [UEC Food datasets](http://foodcam.mobi/dataset.html). A noteworthy point is that the CNNs were trained on Google Colaboratory using the TPU, and it greatly facilitated fast experimental iteration to evaluate various image augmentation and fine-tuning schemes.
## 2. How to use
A typical process flow may involve the following:
- Download the [UEC Food datasets](http://foodcam.mobi/dataset.html)
- Run **create_tfrecords.py** to extract the ground-truth crops for each image in the dataset, and serialize it into *.tfrecords* format to simplify and speed-up the training loop
- Run **calc_channel_means.py** to compute the per-pixel channel means to normalize and "center" the pixel values. Save the means in **helper_funcs.py**
- Create a Google Cloud Storage instance, and upload the *.tfrecords* files into it
- Open **main_training_loop.ipynb** on Google Colaboratory, and connect to the TPU server
- Modify the code cells to point to your Google Cloud Storage instance on which the *.tfrecords* files are located
- On the TPU instance, upload **helper_funcs.py** and **food_classifier_models.py**
- Execute the various cells in **main_training_loop.ipynb**, and run the training loop
- Open up the Google Drive directory in which the results are saved, and use them for whatever purpose you wish
