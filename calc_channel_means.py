'''
	This script calculates the per-channel means across the training datasets. Since 
	these datasets have no validation sets, official splits have been provided for 
	experimentation. To allow us to use each of the 5 splits as validation split should
	we desire, 5 different per-channel means are calculated -- one corresponding to each 
	training split omitted as a validation split. Since there are 3 channels for which 
	the means are calculated (R, G, and B), there are 15 output values in total

	In particular, this script expects the UEC Food dataset to be serialized in .tfrecords 
	format, as defined in create_tfrecords.py. The means will be printed to the command 
	line from which this script is run, and it expects the user to manually record them
'''

import tensorflow as tf
import tensorflow.contrib.eager as tfe 	# used for convenient calls to certain methods

import numpy as np

#import helper_funcs as helper


##  Enable the eager execution API of tensorflow
tf.enable_eager_execution()



##################################### CONSTANTS ####################################

DATASET = 0		# 0 for UEC100, 1 for UEC256

''' This group of constants below is used to set the base directory for the tfrecords files '''
TFRECORD_BASE = 'training_tfrecords'
UEC100_TFRECORD_DIR = '/uecfood100_split'
UEC256_TFRECORD_DIR = '/uecfood256_split'
UEC_TFRECORD_SPLIT = ['/val0.tfrecords', '/val1.tfrecords', '/val2.tfrecords', '/val3.tfrecords', '/val4.tfrecords']


''' The group below defines a set of constants whose values are assigned based on the dataset (UEC100 or UEC256). It's to maintain one point of change '''
TFRECORD_DIR_BASE = None

''' This if statement assigns constant values (typically for paths) based on whether we're training on UEC100 or 256 '''
if (DATASET == 0):
	TFRECORD_DIR_BASE = TFRECORD_BASE + UEC100_TFRECORD_DIR
elif (DATASET == 1):
	TFRECORD_DIR_BASE = TFRECORD_BASE + UEC256_TFRECORD_DIR

############ END ############


def parse_example(example_proto):
	features = { "image_raw": tf.FixedLenFeature((), tf.string, default_value=""), 
				"label": tf.FixedLenFeature((), tf.int64, default_value=0),
				"height": tf.FixedLenFeature((), tf.int64, default_value=0),
				"width": tf.FixedLenFeature((), tf.int64, default_value=0)}

	parsed_features = tf.parse_single_example(example_proto, features)		# returns a dictionary mapping the key of features to its value

	height = tf.cast(parsed_features['height'], tf.int32)		# get image height
	width = tf.cast(parsed_features['width'], tf.int32)		# get image width
	label = tf.cast(parsed_features['label'], tf.int32)			# get the label as an int32 tensor
	image = tf.reshape(tf.decode_raw(parsed_features['image_raw'], tf.uint8), [height, width, 3])	# returns the image tensor with [height, width, channels=3]

	# we only need the image to calculate the means
	return image


if __name__ == "__main__":
	# Main code goes here
	
	list_pix_vals_by_chan_by_set = []	# there will be 5 entries, one for each dataset split. Each entry will contain 3 arrays for all R, G, and B pixel values

	for split_path in UEC_TFRECORD_SPLIT:
		full_split_path = TFRECORD_DIR_BASE + split_path

		split_dataset = tf.data.TFRecordDataset(full_split_path)
		split_dataset = split_dataset.map(parse_example, num_parallel_calls=4)	# each iteration now provides the groundtruth image

		list_pixel_vals_by_channel = [[],[],[]]

		for image in split_dataset:
			# image is a tensor of shape [height, width, 3]
			image = image.numpy()	# convert image tensor to numpy array
			'''We want to avoid using numpy's append function since it copies arrays'''

			# Get all pixels by channel, and flatten to ensure no issue with shape
			for i in range(3):
				channel_pixels = image[...,i].flatten()		# get all pixels in channel and flatten the 2D result
				list_pixel_vals_by_channel[i].append(channel_pixels)	# python append. Flatten again later

		for j in range(3):
			# Right now, the list is split by channel by image. Thus, for each channel, flatten again
			list_pixel_vals_by_channel[j] = np.concatenate(list_pixel_vals_by_channel[j], axis=None)

		#print("Shape: " + str(list_pixel_vals_by_channel[0].shape) + ', ' + str(list_pixel_vals_by_channel[1].shape) + ', ' + str(list_pixel_vals_by_channel[2].shape))
		list_pix_vals_by_chan_by_set.append(list_pixel_vals_by_channel)		# list will contain 5 entries, each containing the pixel vals for each set by channel
	

	# At this point, we can calculate the 5 means
	per_channel_means = []	# we expect each entry appended to this to be of form "[R_mean, G_mean, B_mean]"

	for i in range(len(list_pix_vals_by_chan_by_set)):
		mean_by_channel = []

		list_of_indices = [j for j in range(5)]
		list_of_indices.remove(i)	# returns indices without i. There will always be 4 in total

		for j in range(3):
			# Iterate for each of R, G, and B. Create array, calculate mean, discard values (to be memory-efficient)
			training_set_channel_pixels = np.concatenate([list_pix_vals_by_chan_by_set[list_of_indices[0]][j], 
				list_pix_vals_by_chan_by_set[list_of_indices[1]][j], 
				list_pix_vals_by_chan_by_set[list_of_indices[2]][j],
				list_pix_vals_by_chan_by_set[list_of_indices[3]][j]], axis=None)

			#print("Length training_set_channel_pixels: ", training_set_channel_pixels.shape)

			channel_mean = np.mean(training_set_channel_pixels)
			training_set_channel_pixels = None	# immediately clear from memory

			print("Channel mean: ", channel_mean)

			mean_by_channel.append(channel_mean)

		# At this point in the code, all 3 channels should have had their means calculated and appended into mean_by_channel in RGB order
		per_channel_means.append(mean_by_channel)

	print("Per channel means: ", per_channel_means)