'''
	This script is used to plot the accuracy-epoch curves for various training results.
	It was primarily a convenience script for myself, and as such, it is a bit messy. An 
	easy way to think of this script is that it reads a csv file containing the validation 
	accuracy at the corresponding epoch, and simply plots it

	The various functions were hardcoded for my particular needs; it may be useful to think 
	of this as a reference script with which you may implement your own plotting functions
	based on your own needs.
'''

# Plot CSVs
import numpy as np
import matplotlib.pyplot as plt
import csv

path = 'results'
folders = {
	'tl':'/transfer_learning', 
	'aug':'/image_aug_transfer',
	'ft':'/fine_tune_amounts',
	'final':'/final_acc'
}
what_to_plot = 'tl'
# what_to_plot = 'aug'
# what_to_plot = 'ft'
# what_to_plot = 'final'

full_base_path = path + folders[what_to_plot]

def plot_tl():
	file_paths = ['/aug1_dense.csv','/notrain_dense.csv']

	with open(full_base_path + file_paths[0]) as csv_file:
		reader = csv.reader(csv_file)
		pretrain_results = dict(reader)

	with open(full_base_path + file_paths[1]) as csv_file:
		reader = csv.reader(csv_file)
		notrain_results = dict(reader)

	for key, val in pretrain_results.items():
		pretrain_results[key] = eval(val)

	for key, val in notrain_results.items():
		notrain_results[key] = eval(val)
	# Plot losses on one graph, accuracy on another
	# To explain, used aug_1 as it was the most varied of the augmentation schemes, which is important for initial training to make sure it doesn't overfit early on
	epochs = range(len(notrain_results['loss']))

	plt.figure(1)
	ax = plt.subplot(111)

	ax.plot(epochs, pretrain_results['categorical_accuracy'], 'b', label='Pre-Trained Training Acc')
	ax.plot(epochs, notrain_results['categorical_accuracy'], 'r', label='Random-Weights Training Acc')

	ax.plot(epochs, pretrain_results['val_categorical_accuracy'], 'b--', label='Pre-Trained Validation Acc')
	ax.plot(epochs, notrain_results['val_categorical_accuracy'], 'r--', label='Random-Weights Validation Acc')

	# Shrink current axis by 20%
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

	plt.xlabel('Epoch')
	plt.ylabel('Accuracy (Ratio)')
	plt.title('Pre-Trained vs Random-Weight DenseNet Accuracies')
	ax.legend(bbox_to_anchor=(1.02,0.5), loc="center left", fontsize='small')


	plt.figure(2)

	plt.plot(epochs, pretrain_results['loss'], 'b', label='Pre-Trained Training Loss')
	plt.plot(epochs, notrain_results['loss'], 'r', label='Random-Weights Training Loss')

	plt.plot(epochs, pretrain_results['val_loss'], 'b--', label='Pre-Trained Validation Loss')
	plt.plot(epochs, notrain_results['val_loss'], 'r--', label='Random-Weights Validation Loss')

	plt.xlabel('Epoch')
	plt.ylabel('Accuracy (Ratio)')
	plt.title('Pre-Trained vs Random-Weight DenseNet Losses')
	plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", fontsize='small')

	plt.show()


def plot_aug():
	dataset_paths = ['/uec100', '/uec256']
	file_paths = ['/dense_aug1.csv',
		'/dense_aug2.csv',
		'/dense_aug3.csv',
		'/dense_noaug.csv',
		'/mob_aug1.csv',
		'/mob_aug2.csv',
		'/mob_aug3.csv',
		'/mob_noaug.csv']

	for i in range(len(dataset_paths)):
		folder_path = full_base_path + dataset_paths[i]

		with open(folder_path + file_paths[0]) as csv_file:
			reader = csv.reader(csv_file)
			dense_aug1 = dict(reader)
		with open(folder_path + file_paths[1]) as csv_file:
			reader = csv.reader(csv_file)
			dense_aug2 = dict(reader)
		with open(folder_path + file_paths[2]) as csv_file:
			reader = csv.reader(csv_file)
			dense_aug3 = dict(reader)
		with open(folder_path + file_paths[3]) as csv_file:
			reader = csv.reader(csv_file)
			dense_noaug = dict(reader)
		with open(folder_path + file_paths[4]) as csv_file:
			reader = csv.reader(csv_file)
			mob_aug1 = dict(reader)
		with open(folder_path + file_paths[5]) as csv_file:
			reader = csv.reader(csv_file)
			mob_aug2 = dict(reader)
		with open(folder_path + file_paths[6]) as csv_file:
			reader = csv.reader(csv_file)
			mob_aug3 = dict(reader)
		with open(folder_path + file_paths[7]) as csv_file:
			reader = csv.reader(csv_file)
			mob_noaug = dict(reader)

		for key, val in dense_aug1.items():
			dense_aug1[key] = eval(val)
		for key, val in dense_aug2.items():
			dense_aug2[key] = eval(val)
		for key, val in dense_aug3.items():
			dense_aug3[key] = eval(val)
		for key, val in dense_noaug.items():
			dense_noaug[key] = eval(val)
		for key, val in mob_aug1.items():
			mob_aug1[key] = eval(val)
		for key, val in mob_aug2.items():
			mob_aug2[key] = eval(val)
		for key, val in mob_aug3.items():
			mob_aug3[key] = eval(val)
		for key, val in mob_noaug.items():
			mob_noaug[key] = eval(val)
	
		epochs = range(len(dense_aug1['loss']))

		plt.figure(1 + (i*3))
		ax = plt.subplot(111)

		ax.plot(epochs, dense_noaug['categorical_accuracy'], 'b', label='No Augment Training Acc')
		ax.plot(epochs, dense_noaug['val_categorical_accuracy'], 'b--', label='No Augment Validation Acc')

		ax.plot(epochs, dense_aug1['categorical_accuracy'], 'r', label='Augment #1 Training Acc')
		ax.plot(epochs, dense_aug1['val_categorical_accuracy'], 'r--', label='Augment #1 Validation Acc')

		ax.plot(epochs, dense_aug2['categorical_accuracy'], 'g', label='Augment #2 Training Acc')
		ax.plot(epochs, dense_aug2['val_categorical_accuracy'], 'g--', label='Augment #2 Validation Acc')

		ax.plot(epochs, dense_aug3['categorical_accuracy'], 'y', label='Augment #3 Training Acc')
		ax.plot(epochs, dense_aug3['val_categorical_accuracy'], 'y--', label='Augment #3 Validation Acc')

		# Shrink current axis by 20%
		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

		plt.xlabel('Epoch')
		plt.ylabel('Accuracy (Ratio)')
		if (i == 0):
			plt.title('DenseNet Accuracies - UEC100')
		else:
			plt.title('DenseNet Accuracies - UEC256')
		
		ax.legend(bbox_to_anchor=(1.02,0.5), loc="center left", fontsize='small')


		plt.figure(2 + (i*3))
		ax2 = plt.subplot(111)

		ax2.plot(epochs, mob_noaug['categorical_accuracy'], 'b', label='No Augment Training Acc')
		ax2.plot(epochs, mob_noaug['val_categorical_accuracy'], 'b--', label='No Augment Validation Acc')

		ax2.plot(epochs, mob_aug1['categorical_accuracy'], 'r', label='Augment #1 Training Acc')
		ax2.plot(epochs, mob_aug1['val_categorical_accuracy'], 'r--', label='Augment #1 Validation Acc')

		ax2.plot(epochs, mob_aug2['categorical_accuracy'], 'g', label='Augment #2 Training Acc')
		ax2.plot(epochs, mob_aug2['val_categorical_accuracy'], 'g--', label='Augment #2 Validation Acc')

		ax2.plot(epochs, mob_aug3['categorical_accuracy'], 'y', label='Augment #3 Training Acc')
		ax2.plot(epochs, mob_aug3['val_categorical_accuracy'], 'y--', label='Augment #3 Validation Acc')

		# Shrink current axis by 20%
		box = ax2.get_position()
		ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])

		plt.xlabel('Epoch')
		plt.ylabel('Accuracy (Ratio)')
		if (i == 0):
			plt.title('MobNetV2 Accuracies - UEC100')
		else:
			plt.title('MobNetV2 Accuracies - UEC256')
		
		ax2.legend(bbox_to_anchor=(1.02,0.5), loc="center left", fontsize='small')

	plt.show()


def plot_ft():
	dataset_paths = ['/uec100_dense', '/uec256_dense']
	uec100_paths = ['/noaug.csv', '/noaug_pt2.csv', '/noaug_pt5.csv']
	uec256_paths = ['/noaug.csv', '/noaug_pt6.csv', '/noaug_pt8.csv', '/noaug_pt25.csv']

	# Plot UEC100 stuff first
	folder_path = full_base_path + dataset_paths[0]

	with open(folder_path + uec100_paths[0]) as csv_file:
		reader = csv.reader(csv_file)
		noaug = dict(reader)
	with open(folder_path + uec100_paths[1]) as csv_file:
		reader = csv.reader(csv_file)
		noaug2 = dict(reader)
	with open(folder_path + uec100_paths[2]) as csv_file:
		reader = csv.reader(csv_file)
		noaug5 = dict(reader)

	for key, val in noaug.items():
		noaug[key] = eval(val)
	for key, val in noaug2.items():
		noaug2[key] = eval(val)
	for key, val in noaug5.items():
		noaug5[key] = eval(val)

	epochs = range(len(noaug['loss']))

	plt.figure(1)
	ax = plt.subplot(111)

	ax.plot(epochs, noaug['val_categorical_accuracy'], 'b--', label='1.0 of Layers, Validation Acc')
	ax.plot(epochs, noaug2['val_categorical_accuracy'], 'r--', label='.75 of Layers, Validation Acc')
	ax.plot(epochs, noaug5['val_categorical_accuracy'], 'g--', label='.5 of Layers, Validation Acc')

	# Shrink current axis by 20%
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

	plt.xlabel('Epoch')
	plt.ylabel('Accuracy (Ratio)')
	plt.title('DenseNet Fine-Tuning Amounts - UEC100')
	
	ax.legend(bbox_to_anchor=(1.02,0.5), loc="center left", fontsize='small')


	# Plot UEC256
	folder_path = full_base_path + dataset_paths[1]

	with open(folder_path + uec256_paths[0]) as csv_file:
		reader = csv.reader(csv_file)
		noaug = dict(reader)
	with open(folder_path + uec256_paths[1]) as csv_file:
		reader = csv.reader(csv_file)
		noaug6 = dict(reader)
	with open(folder_path + uec256_paths[2]) as csv_file:
		reader = csv.reader(csv_file)
		noaug8 = dict(reader)
	with open(folder_path + uec256_paths[3]) as csv_file:
		reader = csv.reader(csv_file)
		noaug25 = dict(reader)

	for key, val in noaug.items():
		noaug[key] = eval(val)
	for key, val in noaug6.items():
		noaug6[key] = eval(val)
	for key, val in noaug8.items():
		noaug8[key] = eval(val)
	for key, val in noaug25.items():
		noaug25[key] = eval(val)

	epochs = range(len(noaug['loss']))

	plt.figure(2)
	ax2 = plt.subplot(111)

	ax2.plot(epochs, noaug['val_categorical_accuracy'], 'b--', label='1.0 of Layers, Validation Acc')
	ax2.plot(epochs, noaug25['val_categorical_accuracy'], 'r--', label='.75 of Layers, Validation Acc')
	ax2.plot(epochs, noaug6['val_categorical_accuracy'], 'g--', label='.5 of Layers, Validation Acc')
	ax2.plot(epochs, noaug8['val_categorical_accuracy'], 'y--', label='.25 of Layers, Validation Acc')

	# Shrink current axis by 20%
	box = ax2.get_position()
	ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])

	plt.xlabel('Epoch')
	plt.ylabel('Accuracy (Ratio)')
	plt.title('DenseNet Fine-Tuning Amounts - UEC256')
	
	ax2.legend(bbox_to_anchor=(1.02,0.5), loc="center left", fontsize='small')

	plt.show()


def plot_final():
	uec100_path = full_base_path + '/uec100'
	file_paths = ['/dense.csv','/dense_ws.csv','/mob.csv','/mob_ws.csv']
	uec256_file_paths = ['/dense256.csv','/dense256_ws.csv','/mob256.csv','/mob256_ws.csv']

	with open(uec100_path + file_paths[0]) as csv_file:
		reader = csv.reader(csv_file)
		dense = dict(reader)
	with open(uec100_path + file_paths[1]) as csv_file:
		reader = csv.reader(csv_file)
		dense_ws = dict(reader)
	with open(uec100_path + file_paths[2]) as csv_file:
		reader = csv.reader(csv_file)
		mob = dict(reader)
	with open(uec100_path + file_paths[3]) as csv_file:
		reader = csv.reader(csv_file)
		mob_ws = dict(reader)
	with open(full_base_path + uec256_file_paths[0]) as csv_file:
		reader = csv.reader(csv_file)
		dense256 = dict(reader)
	with open(full_base_path + uec256_file_paths[1]) as csv_file:
		reader = csv.reader(csv_file)
		dense_ws256 = dict(reader)
	with open(full_base_path + uec256_file_paths[2]) as csv_file:
		reader = csv.reader(csv_file)
		mob256 = dict(reader)
	with open(full_base_path + uec256_file_paths[3]) as csv_file:
		reader = csv.reader(csv_file)
		mob_ws256 = dict(reader)

	for key, val in dense.items():
		dense[key] = eval(val)
	for key, val in dense_ws.items():
		dense_ws[key] = eval(val)
	for key, val in mob.items():
		mob[key] = eval(val)
	for key, val in mob_ws.items():
		mob_ws[key] = eval(val)
	for key, val in dense256.items():
		dense256[key] = eval(val)
	for key, val in dense_ws256.items():
		dense_ws256[key] = eval(val)
	for key, val in mob256.items():
		mob256[key] = eval(val)
	for key, val in mob_ws256.items():
		mob_ws256[key] = eval(val)

	epochs = range(len(dense['loss']))

	plt.figure(1)
	ax = plt.subplot(111)

	# ax.plot(epochs, dense['categorical_accuracy'], 'b', label='DenseNet Training Acc')
	ax.plot(epochs, dense['val_categorical_accuracy'], 'b--', label='DenseNet')

	# ax.plot(epochs, dense_ws['categorical_accuracy'], 'r', label='Wide-Slice DenseNet Training Acc')
	ax.plot(epochs, dense_ws['val_categorical_accuracy'], 'r--', label='Wide-Slice DenseNet')

	# ax.plot(epochs, mob['categorical_accuracy'], 'g', label='MobileNetV2 Training Acc')
	ax.plot(epochs, mob['val_categorical_accuracy'], 'g--', label='MobileNetV2')

	# ax.plot(epochs, mob_ws['categorical_accuracy'], 'y', label='Wide-Slice MobileNetV2 Training Acc')
	ax.plot(epochs, mob_ws['val_categorical_accuracy'], 'y--', label='Wide-Slice MobileNetV2')

	# Shrink current axis by 20%
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

	plt.xlabel('Epoch')
	plt.ylabel('Accuracy (Ratio)')
	plt.title('UEC100 Single-Crop Validation Accuracies')
	
	ax.legend(bbox_to_anchor=(1.02,0.5), loc="center left", fontsize='small')


	plt.figure(2)
	ax2 = plt.subplot(111)

	# ax2.plot(epochs, dense256['categorical_accuracy'], 'b', label='DenseNet Training Acc')
	ax2.plot(epochs, dense256['val_categorical_accuracy'], 'b--', label='DenseNet')

	# ax2.plot(epochs, dense_ws256['categorical_accuracy'], 'r', label='Wide-Slice DenseNet Training Acc')
	ax2.plot(epochs, dense_ws256['val_categorical_accuracy'], 'r--', label='Wide-Slice DenseNet')

	# ax2.plot(epochs, mob256['categorical_accuracy'], 'g', label='MobileNetV2 Training Acc')
	ax2.plot(epochs, mob256['val_categorical_accuracy'], 'g--', label='MobileNetV2')

	# ax2.plot(epochs, mob_ws256['categorical_accuracy'], 'y', label='Wide-Slice MobileNetV2 Training Acc')
	ax2.plot(epochs, mob_ws256['val_categorical_accuracy'], 'y--', label='Wide-Slice MobileNetV2')

	# Shrink current axis by 20%
	box = ax2.get_position()
	ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])

	plt.xlabel('Epoch')
	plt.ylabel('Accuracy (Ratio)')
	plt.title('UEC256 Single-Crop Validation Accuracies')
	
	ax2.legend(bbox_to_anchor=(1.02,0.5), loc="center left", fontsize='small')

	plt.show()



if __name__ == "__main__":

	if what_to_plot == 'tl':
		plot_tl()

	elif what_to_plot == 'aug':
		plot_aug()

	elif what_to_plot == 'ft':
		plot_ft()

	elif what_to_plot == 'final':
		plot_final()

	else:
		print('what_to_plot set to unexpected value: ', what_to_plot)
		sys.exit()
