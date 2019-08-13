###################################################################################
## Helper functions for various scripts, and training procedures
## NOTE THAT THE parse_tfrecord_example FUNCTION NEEDS TO BE CHANGED FOR UEC100 
## AND UEC256. This is an inconvenience that may be patched in a later update.
###################################################################################

import tensorflow as tf
import tensorflow.contrib.eager as tfe 	# used to create Variable
import numpy as np

import csv
import traceback


# Below are the computed per-pixel means by channel for each combination of training sets. In other words: 
# Index 0 contains the mean for training sets 1-4, index 4 for training sets 0-3, etc.
PER_PIXEL_CHANNEL_MEANS = [[147.85033763377476, 123.00073632083487, 90.232759117071808], 
	[148.13072784696206, 123.11814840681936, 90.189210486066798], 
	[148.17377068535109, 123.37010733059971, 90.685267314392235], 
	[148.00018582056654, 123.03650484470955, 90.281080919877795], 
	[147.99504404878493, 123.04823646631419, 90.198134460960318]]

# Below, we choose the one we want to use. We don't average all results as the means calculated from the validation set should be withheld. 
# In a real scenario, this information won't be available before-hand, and thus it shouldn't be included
MEAN_PER_PIXEL_MEANS_BY_CHANNEL = [147.99504404878493, 123.04823646631419, 90.198134460960318]	# means for a validation set with split #5


'''
This function reads the official dataset split file at the specified directory

@param split_dir_base : The base path
@param split_file_dir : The split file name
@return str_arr : An array of strings containing all split file names
'''
def read_dataset_split_file(split_dir_base, split_file_dir):
	file = open(split_dir_base + split_file_dir, 'r')
	str_arr = file.readlines()

	for j in range(len(str_arr)):
		str_arr[j] = str_arr[j][:-1]	# removes the trailing \n. read().split("\n") results in some empty entries, so it's safer to do this

	file.close()
	return str_arr



'''
This function reads the official dataset split files at the specified directories

@param split_dir_base : The base path
@param split_files : A list of split file names
@return input_x_fold_split : An array containing arrays of strings, containing 
all split file names, divided by split
'''
def read_dataset_split_files(split_dir_base, split_files):
	input_x_fold_split = []
	for i in range(len(split_files)):
		split_data = read_dataset_split_file(split_dir_base, split_files[i])
		input_x_fold_split.append(split_data)

	return input_x_fold_split



'''
This function gets bounding box info based on the image directory base. The 
datasets have folders that are numbered and correspond to each of the different 
classes; this function goes into each of these, reads the bounding box csv file 
for each, and places them into a return dictionary

@param image_dir_base : The base path for the dataset images
@param num_classes : The number of classes (100 for UEC100, 256 for UEC256)
@return bb_info_dict : A dictionary containing all bounding box info by image
'''
def get_bounding_box_info(image_dir_base, num_classes):
	path_base = image_dir_base + '/'
	path_end = '/bb_info.txt'

	bb_info_dict = {}

	for i in range(1, num_classes + 1):
		bb_info_dict[i] = {}
		try:
			file_reader = csv.DictReader(open(path_base + str(i) + path_end, 'r'), delimiter = ' ')
			for file_row in file_reader:
				''' Expected keys: img, x1, y1, x2, y2 '''
				img = file_row['img']
				bb_info_dict[i][img] = {}
				bb_info_dict[i][img]['x1'] = int(file_row['x1'])
				bb_info_dict[i][img]['y1'] = int(file_row['y1'])
				bb_info_dict[i][img]['x2'] = int(file_row['x2'])
				bb_info_dict[i][img]['y2'] = int(file_row['y2'])
			
		except Exception: 
			print("Error loading bb_info for category: " + str(i))
			traceback.print_exc()
		
	return bb_info_dict



'''
This function calculates the target height/width based on bounding box info, 
and ensures it doesn't exceed image dimesions -- which can cause errors. 
Another phrasing is that it clips the bounding box coordinates such that 
they always fall within the image dimensions.

@param bb_lower_bound : x1 or y1 of the bounding box
@param bb_upper_bound : x2 or y2 of the bounding box
@param image_dim_val : The width or height of the image
@return target_dim_val : The target width or height that contains the 
ground truth image
'''
# This function calculates the target height/width based on bounding box info, ensuring it doesn't exceed the image dimensions
def get_target_dim_val(bb_lower_bound, bb_upper_bound, image_dim_val):
	if (bb_upper_bound <= image_dim_val):
		target_dim_val = bb_upper_bound - bb_lower_bound
	else:
		target_dim_val = image_dim_val - bb_lower_bound

	return target_dim_val


# This function converts the argument into a format that would be accepted for storing in the tfrecords format
def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))



# This function converts the argument into a format that would be accepted for storing in the tfrecords format
def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


'''
This function parses the .tfrecords file, line-by-line, to extract each 
entry (or example) and acquire the pre-processed image and label. It is 
passed as a map function for the TensorFlow data API.

IMPORTANT: MODIFY THE NUMBER OF ENTRIES FOR THE ONE_HOT LABEL DEPENDING 
ON THE DATASET

@param example_proto : Required argument to be passed as a map function. 
Represents a single line/entry 
@return label, image : The label and image for which the example_proto
corresponds
'''
def parse_tfrecord_example(example_proto):	
	features = { "image_raw": tf.FixedLenFeature((), tf.string, default_value=""), 
				"label": tf.FixedLenFeature((), tf.int64, default_value=0),
				"height": tf.FixedLenFeature((), tf.int64, default_value=0),
				"width": tf.FixedLenFeature((), tf.int64, default_value=0)}

	# returns a dictionary mapping the key of features to its value
	parsed_features = tf.parse_single_example(example_proto, features)

	height = tf.cast(parsed_features['height'], tf.int32)	# get image height
	width = tf.cast(parsed_features['width'], tf.int32)		# get image width
	label = tf.cast(parsed_features['label'], tf.int32)		# get the label as an int32 tensor
	# creates the image tensor with dim [height, width, channels=3]
	image = tf.reshape(tf.decode_raw(parsed_features['image_raw'], tf.uint8), [height, width, 3])

	# centers data around the mean, then divides by 128 to get range of 2 
	image = tf.truediv(tf.subtract(tf.cast(image, tf.float32), MEAN_PER_PIXEL_MEANS_BY_CHANNEL), 128.0)

	label = tf.subtract(label, 1)	# Subtract 1 to normalize to [0, num_classes)
	#label = tf.one_hot(label, 100, dtype=tf.float32)	# facilitates .fit keras method. UEC100
	label = tf.one_hot(label, 256, dtype=tf.float32)	# facilitates .fit keras method. UEC256

	return label, image



'''
This function augments the images based on image augmentation scheme 1 
as proposed in my thesis. It was not very useful, and is left here in 
case you wish to experiment with it. This is also a map function for 
the TensorFlow data API

@param label : Required argument that corresponds to the 
parse_tfrecord_example map function output. 
@param image : Required argument that corresponds to the 
parse_tfrecord_example map function output. 
@return images, labels : A list of images and labels corresponding to 
the augmented entries
'''
def augment_image(label, image):
	'''
		Perform scale jitter idea based on GoogLeNet with smaller side 
		of image resized to 256

		Other desirable transforms include randomly flipping the image, 
		applying mild brightness/contrast transforms, and taking a 
		central crop without distortions except flip
	'''
	labels = []
	images = []

	image = tf.image.random_flip_left_right(image)
	images.append(tf.image.resize_image_with_crop_or_pad(image, 224, 224))
	# To facilitate unbatching, labels is also passed as a list to match 
	# the format of images
	labels.append(label)

	image_shape = tf.shape(image)
	image_height = tf.cast(image_shape[0], dtype=tf.float32)
	image_width = tf.cast(image_shape[1], dtype=tf.float32)

	rescale_factor = tf.divide(tf.constant(256, dtype=tf.float32), tf.minimum(image_height, image_width))

	image_height = tf.cast(tf.multiply(image_height, rescale_factor), dtype=tf.int32)
	image_width = tf.cast(tf.multiply(image_width, rescale_factor), dtype=tf.int32)
	image_dims = tf.stack([image_height, image_width])

	image = tf.image.resize_images(image, image_dims)	# resize image to have smaller dimension 256

	image = tf.image.random_brightness(image, 0.15)		# adds brightness distortion with 0.15 delta
	image = tf.image.random_contrast(image, 0.8, 1.2)	# modify contrast by up to 0.2
	image = tf.image.random_saturation(image, 0.7, 1.3)	# modify saturation by up to 0.3

	# The section below gets center crops at different %s of the image, and scales it up to the original image width and height
	scale_factors = list(np.arange(0.4, 1.0, 0.12))
	boxes = np.zeros((len(scale_factors), 4))

	for i, scale_factor in enumerate(scale_factors):
		x1 = y1 = 0.5 - (0.5 * scale_factor)
		x2 = y2 = 0.5 + (0.5 * scale_factor)
		boxes[i] = [y1, x1, y2, x2]

	crops = tf.image.crop_and_resize([image], boxes=boxes, box_ind=np.zeros(len(scale_factors)), crop_size=image_dims)

	dim_tensor = tf.constant([224, 224, 3])
	for j in range(len(scale_factors)):
		labels.append(label)
		images.append(tf.random_crop(crops[j], dim_tensor))

	# Acquire 3 random crops
	for i in range(3):
		images.append(tf.random_crop(image, dim_tensor))
		labels.append(label)

	#images = tf.image.resize_image_with_crop_or_pad(images, 224, 224)	# resizes all images to 224x224 with crops and pads

	return images, labels


'''
This function augments the images based on image augmentation scheme 2
as proposed in my thesis. It was not very useful, and is left here in 
case you wish to experiment with it. This is also a map function for 
the TensorFlow data API

@param label : Required argument that corresponds to the 
parse_tfrecord_example map function output. 
@param image : Required argument that corresponds to the 
parse_tfrecord_example map function output. 
@return images, labels : A list of images and labels corresponding to 
the augmented entries
'''
# Takes the image and applies desirable transformations for training. Returns same-size images to allow for dataset batching
def augment_image_two(label, image):
	'''
		Rescale smaller side 256, append original image randomly flipped, apply distortions, take random crops
	'''
	labels = []
	images = []

	image = tf.image.random_flip_left_right(image)
	images.append(tf.image.resize_image_with_crop_or_pad(image, 224, 224))
	labels.append(label)	# labels is a list as well to facilitate unbatch method on dataset later

	image_shape = tf.shape(image)
	image_height = tf.cast(image_shape[0], dtype=tf.float32)
	image_width = tf.cast(image_shape[1], dtype=tf.float32)

	rescale_factor = tf.divide(tf.constant(256, dtype=tf.float32), tf.minimum(image_height, image_width))

	image_height = tf.cast(tf.multiply(image_height, rescale_factor), dtype=tf.int32)
	image_width = tf.cast(tf.multiply(image_width, rescale_factor), dtype=tf.int32)
	image_dims = tf.stack([image_height, image_width])

	image = tf.image.resize_images(image, image_dims)	# resize image to have smaller dimension 256

	image = tf.image.random_brightness(image, 0.15)		# adds brightness distortion with 0.15 delta
	image = tf.image.random_contrast(image, 0.8, 1.2)	# modify contrast by up to 0.2
	image = tf.image.random_saturation(image, 0.7, 1.3)	# modify saturation by up to 0.3

	# Acquire 8 random crops. This is to be consistent in terms of num aug images from augment_image method
	dim_tensor = tf.constant([224, 224, 3])
	for i in range(8):
		images.append(tf.random_crop(image, dim_tensor))
		labels.append(label)

	return images, labels


'''
This function augments the images based on image augmentation scheme 3
as proposed in my thesis. It augments a single additional image 
comprising all desired distortions, and was the most beneficial scheme 
in my experiments. This is also a map function for the TensorFlow data API

@param label : Required argument that corresponds to the 
parse_tfrecord_example map function output. 
@param image : Required argument that corresponds to the 
parse_tfrecord_example map function output. 
@return images, labels : A list of images and labels corresponding to 
the augmented entries
'''
def augment_image_three(label, image):
	labels = []
	images = []

	images.append(tf.image.resize_image_with_crop_or_pad(image, 224, 224))
	labels.append(label)	# labels is a list as well to facilitate unbatch method on dataset later

	image = tf.image.random_flip_left_right(image)
	
	image_shape = tf.shape(image)
	image_height = tf.cast(image_shape[0], dtype=tf.float32)
	image_width = tf.cast(image_shape[1], dtype=tf.float32)

	rescale_factor = tf.divide(tf.constant(256, dtype=tf.float32), tf.minimum(image_height, image_width))

	image_height = tf.cast(tf.multiply(image_height, rescale_factor), dtype=tf.int32)
	image_width = tf.cast(tf.multiply(image_width, rescale_factor), dtype=tf.int32)
	image_dims = tf.stack([image_height, image_width])

	image = tf.image.resize_images(image, image_dims)	# resize image to have smaller dimension 256

	image = tf.image.random_brightness(image, 0.15)		# adds brightness distortion with 0.15 delta
	image = tf.image.random_contrast(image, 0.8, 1.2)	# modify contrast by up to 0.2
	image = tf.image.random_saturation(image, 0.7, 1.3)	# modify saturation by up to 0.3

	# The section below gets center crops at different %s of the image, and scales it up to the original image width and height
	scale_factors = list(np.arange(0.4, 1.0, 0.12))
	boxes = np.zeros((len(scale_factors), 4))

	for i, scale_factor in enumerate(scale_factors):
		x1 = y1 = 0.5 - (0.5 * scale_factor)
		x2 = y2 = 0.5 + (0.5 * scale_factor)
		boxes[i] = [y1, x1, y2, x2]

	crops = tf.image.crop_and_resize([image], boxes=boxes, box_ind=np.zeros(len(scale_factors)), crop_size=image_dims)

	dim_tensor = tf.constant([224, 224, 3])
	images.append(tf.random_crop(crops[tf.random_uniform(shape=[], minval=0, maxval=len(scale_factors), dtype=tf.int32)], dim_tensor))
	labels.append(label)

	return images, labels


'''
This is a map function for the validation set. It just resizes the 
validation image to 224x224. It is centrally cropped if larger than 
224, and is padded if smaller

@param label : Required argument that corresponds to the 
parse_tfrecord_example map function output. 
@param image : Required argument that corresponds to the 
parse_tfrecord_example map function output. 
@return image, label : The resized image, and the same label
'''
def resize_validation_image(label, image):
	image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)	# pads up smaller dimension/s to 224

	return image, label


'''
This is an experimental map function for the validation set. 
Instead of central cropping if the larger dimension is greater
than 224, it resizes first to 224 to capture the entire image. 
There was negligible benefit, if any, for this scheme

@param label : Required argument that corresponds to the 
parse_tfrecord_example map function output. 
@param image : Required argument that corresponds to the 
parse_tfrecord_example map function output. 
@return image, label : The resized image, and the same label
'''
def resize_validation_image_two(label, image):
	# Bring larger dimension to 224 if it's larger than 224. This is to capture the whole image
	image_shape = tf.shape(image)
	image_height = tf.cast(image_shape[0], dtype=tf.float32)
	image_width = tf.cast(image_shape[1], dtype=tf.float32)

	rescale_factor = tf.divide(tf.constant(224, dtype=tf.float32), tf.maximum(image_height, image_width))
	image_height = tf.cast(tf.multiply(image_height, rescale_factor), dtype=tf.int32)
	image_width = tf.cast(tf.multiply(image_width, rescale_factor), dtype=tf.int32)
	image_dims = tf.stack([image_height, image_width])

	image = tf.cond(
		tf.less(rescale_factor, 1), 
		lambda: tf.image.resize_images(image, image_dims), 
		lambda: tf.identity(image))
	
	image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)	# pads up smaller dimension/s to 224

	return image, label