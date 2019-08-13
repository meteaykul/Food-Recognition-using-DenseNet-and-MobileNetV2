'''
	This script parses the UEC Food datasets, pre-processes each image, and serializes them in 
	.tfrecords format. This is done to speed-up the training process as the file format is 
	optimized for TensorFlow training loops; instead of loading the image and pre-processing
	it, or storing the pre-processed images in memory, it simply parses the .tfrecords file 
	when needed for each example image.

	To use these files in conjunction with Google Colaboratory (for TPU access), a Google Cloud 
	account is needed to store the files. It is possible to upload these files on the TPU 
	instance, but it is extremely slow -- especially given the large size of the serialized 
	dataset splits.
'''

import tensorflow as tf
import tensorflow.contrib.eager as tfe 	# used for convenient calls to certain methods

import numpy as np

import helper_funcs as helper

import sys

##  Enable the eager execution API of tensorflow
tf.enable_eager_execution()

## CONSTANTS ##
DATASET = 1		# 0 = UEC100, 1 = UEC256. Defined to help minimize the number of constants to change when creating uec100 or 256 files

UEC100_DIR_BASE = 'UECFOOD100'
UEC256_DIR_BASE = 'UECFOOD256'

UEC100_SPLIT_DIR_BASE = 'uecfood100_split'
UEC256_SPLIT_DIR_BASE = 'uecfood256_split'

UEC100_SPLIT_FILES = ['/val0.txt', '/val1.txt', '/val2.txt', '/val3.txt', '/val4.txt']
UEC256_SPLIT_FILES = ['/val0.txt', '/val1.txt', '/val2.txt', '/val3.txt', '/val4.txt']
UEC_TFRECORD_SPLIT = ['/val0.tfrecords', '/val1.tfrecords', '/val2.tfrecords', '/val3.tfrecords', '/val4.tfrecords']

''' 
For reference, UEC100 has the following number of samples in each split: 
val0 = 2913
val1 = 2894
val2 = 2869
val3 = 2849
val4 = 2836

UEC256: 
val0 = 5754
val1 = 5708
val2 = 5659
val3 = 5611
val4 = 5548
'''

TFRECORD_BASE = 'training_tfrecords'
UEC100_TFRECORD_DIR = '/uecfood100_split'
UEC256_TFRECORD_DIR = '/uecfood256_split'

IMAGE_DIR_BASE = None
NUM_CLASSES = None
TFRECORD_DIR_BASE = None
SPLIT_DIR_BASE = None
SPLIT_FILES = None

if (DATASET == 0):
	SPLIT_DIR_BASE = UEC100_SPLIT_DIR_BASE
	SPLIT_FILES = UEC100_SPLIT_FILES
	IMAGE_DIR_BASE = UEC100_DIR_BASE
	NUM_CLASSES = 100
	TFRECORD_DIR_BASE = TFRECORD_BASE + UEC100_TFRECORD_DIR
elif (DATASET == 1):
	SPLIT_DIR_BASE = UEC256_SPLIT_DIR_BASE
	SPLIT_FILES = UEC256_SPLIT_FILES
	IMAGE_DIR_BASE = UEC256_DIR_BASE
	NUM_CLASSES = 256
	TFRECORD_DIR_BASE = TFRECORD_BASE + UEC256_TFRECORD_DIR

## END CONSTANT DEFINITIONS ##

if __name__ == "__main__":

	'''Get all image bounding box info.'''
	bb_info = helper.get_bounding_box_info(IMAGE_DIR_BASE, NUM_CLASSES)		# returns dict mapping class ID : image ID : x1, y1, x2, y2

	for i in range(len(SPLIT_FILES)):
		split_dir = SPLIT_FILES[i]

		'''Load official split. Do one at a time to reduce memory usage'''
		data_split_info = helper.read_dataset_split_file(SPLIT_DIR_BASE, split_dir)	# returns list of image path

		''' Create tfrecords filename corresponding to this split '''
		filename = TFRECORD_DIR_BASE + UEC_TFRECORD_SPLIT[i]

		print('\n\nWriting', filename)
		print("Number of images to write: %d \n\n" % (len(data_split_info)) )

		''' Create a tfrecordwriter object for writing to the file '''
		filewriter = tf.python_io.TFRecordWriter(filename)

		''' Begin looping over all file names'''
		for j in range(len(data_split_info)):

			if (j % 500 == 0):
				# Print statement to provide feedback on progress. Prints every 500 entries to avoid flooding the console
				print("Processing image number %d of %d \n" % (j, len(data_split_info)) )

			image_path = data_split_info[j]		# String file path. It is of format UECFOODxxx/im_class/im_id.jpg

			image_string = tf.read_file(image_path)		# read the entire file 
			image_decoded = tf.image.decode_jpeg(image_string, channels = 3)	# image with height, width, num_chans shape
			
			image_class, image_id = image_path[11:].split('/')		# extract image class and ID. It is of format im_class/im_id.jpg. Thus, a split seperates the 2
			image_class = int(image_class)	# The bb_info dict has this key stored as an int
			image_id = image_id[:-4]	# truncates the .jpg portion of the name
			image_bb_info = bb_info[image_class][image_id]	# dict with keys x1, y1, x2, y2, representing bounding box coordinates

			''' Clip target image dimensions when necessary; some bounding box info exceeds image dimensions, causing errors....'''
			image_height = image_decoded.shape[0]
			image_width = image_decoded.shape[1]

			target_height = helper.get_target_dim_val(image_bb_info['y1'], image_bb_info['y2'], image_height)
			target_width = helper.get_target_dim_val(image_bb_info['x1'], image_bb_info['x2'], image_width)

			image_bb_cropped = tf.image.crop_to_bounding_box(image_decoded, image_bb_info['y1'], image_bb_info['x1'], target_height, target_width)
			image_bb_cropped_string = image_bb_cropped.numpy().tostring()

			''' With the image_class (ie. the label) and the cropped image, we have what we need to create tfrecords entries '''
			tfrecords_example_entry = tf.train.Example(features=tf.train.Features(feature={
				'label': helper._int64_feature(image_class),
				'height': helper._int64_feature(target_height),
				'width': helper._int64_feature(target_width),
				'image_raw': helper._bytes_feature(image_bb_cropped_string)}))

			filewriter.write(tfrecords_example_entry.SerializeToString())
	filewriter.close()
	sys.exit()