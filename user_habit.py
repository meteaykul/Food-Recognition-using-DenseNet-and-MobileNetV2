from __future__ import absolute_import, division, print_function

import os
import sys

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import traceback	# for error tracebacks
import csv
import ast

import food_classifier_models as models

# ___________________________________________________________________________________

MEAN_PER_PIXEL_MEANS_BY_CHANNEL = [147.99504404878493, 123.04823646631419, 90.198134460960318]	# means for a validation set with split #5
PATH_TO_DIET = 'sample_diet'
USR_HBT_COEFF = 0.1

def load_diet_images():

	dishes_by_day_meal = {}

	for i in range(7):
		day = 'day' + str(i+1)
		folder_path_daily = PATH_TO_DIET + '/' + day

		dishes_by_day_meal[day] = {}

		# In folder_path are two more folders titled meal1 and meal2. Inside each are 2 images.
		for j in range(2):
			meal = 'meal' + str(j+1)
			meal_path = folder_path_daily + '/' + meal
			image_names = os.listdir(meal_path)

			dishes_by_day_meal[day][meal] = []

			for image_name in image_names:
				image_string = tf.read_file(meal_path + '/' + image_name)		# read the entire file 
				image = tf.image.decode_jpeg(image_string, channels = 3)	# image with height, width, num_chans shape
				image = tf.truediv(tf.subtract(tf.cast(image, tf.float32), MEAN_PER_PIXEL_MEANS_BY_CHANNEL), 128.0)	# pre-process
				label = int(image_name[:-4]) - 1		# remove .jpg and normalize to 0-255 from 1-256

				image_height = tf.cast(image.shape[0], dtype=tf.float32)
				image_width = tf.cast(image.shape[1], dtype=tf.float32)

				if image_height <= image_width:
					scale_factor = tf.divide(224, image_height)
				else:
					scale_factor = tf.divide(224, image_width)

				image_height = scale_factor * image_height
				image_width = scale_factor * image_width

				image = tf.image.resize_images(image, [image_height, image_width])
				image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)
				image = tf.expand_dims(image, 0)

				dic = {}
				dic['image'] = image
				dic['label'] = label
				dishes_by_day_meal[day][meal].append(dic)

	return dishes_by_day_meal


def get_diet_predictions(model, dishes_by_day_meal):

	predictions = {}

	if model == 'densenet':
		model = models.DenseNet(None, 256, False)	# No batch size, 256 classes, no wideslice
		model.load_weights('dense256weights.h5')	# should be the 256 version
	
	elif model == 'mobv2':
		model = models.MobNetVTwo(None, 256, False)	# No batch size, 256 classes, no wideslice
		model.load_weights('mob256weights.h5')	# should be the 256 version

	for day in dishes_by_day_meal: 
		predictions[day] = {}

		for meal in dishes_by_day_meal[day]:
			predictions[day][meal] = []

			for dish in dishes_by_day_meal[day][meal]:
				image = dish['image']
				label = dish['label']

				#plt.imshow(tf.squeeze(image))
				#plt.show()

				# Prediction has an extra dimension for batch size of 1. Should be removed
				prediction = tf.nn.softmax(tf.squeeze(model(image)))

				top_predics, indices = tf.nn.top_k(prediction, 10)
				combined = [list(entry) for entry in zip(indices.numpy(), top_predics.numpy())]

				# print("Prediction: ", prediction)
				# print("Top predics: ", top_predics)
				# print("Indices: ", indices)
				# print("Combined: ", combined)
				# print("Label: ", label)

				dic = {}
				dic['predics'] = combined
				dic['label'] = label
				predictions[day][meal].append(dic)

	# Predictions should be saved to a csv
	return predictions


def save_predics():
	dishes_by_day_meal = load_diet_images()

	predics = get_diet_predictions('densenet', dishes_by_day_meal)

	with open(PATH_TO_DIET + '/predics_dense.csv', 'w', newline='') as csv_file:	# saves training accuracy
		writer = csv.writer(csv_file)
		for key, value in predics.items():
			writer.writerow([key, value])
	csv_file.close()

	predics = None	# clear memory

	predics = get_diet_predictions('mobv2', dishes_by_day_meal)

	with open(PATH_TO_DIET + '/predics_mob.csv', 'w', newline='') as csv_file:	# saves training accuracy
		writer = csv.writer(csv_file)
		for key, value in predics.items():
			writer.writerow([key, value])
	csv_file.close()


def get_user_habit_info(predics):
	user_habit_info = {}

	for day, meals in predics.items(): 
		for meal, dishes in meals.items():
			for dish in dishes:
				label = int(dish['label'])

				if not label in user_habit_info:
					user_habit_info[label] = {}

				# predics accounts for one week. Sample diet is over a month. All values should be multiplied by 4
				if not meal in user_habit_info[label]:
					user_habit_info[label][meal] = {}

				keys_to_check = ['freq', day]

				for key in keys_to_check:
					if key in user_habit_info[label][meal]:
						user_habit_info[label][meal][key] += 4
					else:
						user_habit_info[label][meal][key] = 4

	return user_habit_info


def get_user_habit_predic(top_5_labels, top_5_sfmx, user_habit_info, day, meal, coeff):
	# First, top_5 must be altered based on user_habit_info. To do so, user_habit_info needs to be pulled for the corresponding label, then 
	# the sfmx value in the corresponding index altered accordingly

	# Following this, the top_5_sfmx must be iterated over to see where the maximum index is, and the corresponding label retrieved as the guess
	user_top_5_sfmx = []
	num_diet_samples = 1#4 * 7 * 4	# 4 meals a day, 7 days a week, over 4 weeks
	for i in range(len(top_5_labels)):
		label = int(top_5_labels[i])

		if label in user_habit_info:
			label_info = user_habit_info[label]
			day_and_meal_freq, meal_freq, other_meal_freq = 0.0, 0.0, 0.0

			for prev_meal, prev_meal_info in label_info.items():
				if meal == prev_meal:
					# we want to iterate over all days if the meals match
					for prev_day in prev_meal_info:
						if prev_day == day:
							day_and_meal_freq = prev_meal_info[day]	# if day matches a previous day, then set day_and_meal_freq
						else:
							meal_freq += prev_meal_info[prev_day]	# if days do not match, the meal frequency should increment instead
				else:
					other_meal_freq = prev_meal_info['freq']	# if the meals do not match, then freq is simply the other_meal_freq

			# Have all 3 frequencies at this point. Compute the coefficient for tweaking the sfmx value
			user_top_5_sfmx.append(top_5_sfmx[i] * (1 + coeff/num_diet_samples*((0.8 * day_and_meal_freq) + (0.15 * meal_freq) + (0.05 * other_meal_freq))))
		else:
			user_top_5_sfmx.append(top_5_sfmx[i])

	# Get index corresponding to largest softmax value
	max_index, max_val = -1, -1
	
	for index, sfmx_val in enumerate(user_top_5_sfmx):
		if sfmx_val > max_val:
			max_index = index
			max_val = sfmx_val

	return top_5_labels[max_index]



if __name__ == "__main__":
	tf.enable_eager_execution()
	print("TF eager execution: %s" % tf.executing_eagerly())	# checks and prints if tf is executing eagerly, which is necessary for following code

	#save_predics()

	relative_file_paths = ['/predics_dense.csv','/predics_mob.csv']

	for i in range(len(relative_file_paths)):

		with open(PATH_TO_DIET + relative_file_paths[i]) as csv_file:
			reader = csv.reader(csv_file)
			predics = dict(reader)

		for key, val in predics.items():
			predics[key] = eval(val)

		user_habit_info = get_user_habit_info(predics)

		num_samples, top_1, top_5, top_10 = 0, 0, 0, 0

		base_context_biases = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]	# change to 5-50 when dividing by num_samples
		user_habit_top_1s = [0 for _ in range(10)]

		for day, meals in predics.items(): 
			for meal, dishes in meals.items():
				for dish in dishes:
					predics = dish['predics']
					label = dish['label']

					# unzips the list, then re-creates the separated lists from the iterator
					predic_labels = list(zip(*predics))

					top_5_labels = predic_labels[0][:5]
					top_5_sfmx = predic_labels[1][:5]

					for j, context_bias in enumerate(base_context_biases):
						user_habit_predic = get_user_habit_predic(top_5_labels, top_5_sfmx, user_habit_info, day, meal, context_bias)
						if user_habit_predic == label:
							# print("User habit correct: ", day + ' ' + meal + ' ' + str(label))
							user_habit_top_1s[j] += 1
						# else:
							# print("User habit INCORRECT: ", day + ' ' + meal + ' ' + str(label))

					num_samples += 1

					try:
						predic_index = predic_labels[0].index(label)
						# print('Correct for: ', day + ' ' + meal + ' ' + str(label))
						print('Index: ', predic_index)
						top_10 += 1

						if predic_index == 0:
							top_1 += 1
						if predic_index < 5:
							top_5 += 1
					except:
						# print('Incorrect for: ', day + ' ' + meal + ' ' + str(label))
						# Not in any of the guesses
						continue
					

		print('\n####################\n####################')

		if i == 0:
			print('DenseNet Top Accs')
		elif i == 1:
			print('MobNetV2 Top Accs')

		print('Top 1: {:.3%}'.format(top_1/num_samples))
		print('Top 5: {:.3%}'.format(top_5/num_samples))
		print('Top 10: {:.3%}'.format(top_10/num_samples))
		print('\n####################\n####################')
		print('Coefficients: ', base_context_biases)
		print('Accuracies: ', ['{:.3%}'.format(user_habit_top_1/num_samples) for user_habit_top_1 in user_habit_top_1s])
		print('\n####################\n####################')