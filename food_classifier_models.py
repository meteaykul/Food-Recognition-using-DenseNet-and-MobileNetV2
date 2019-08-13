###################################################################################
## This file contains functions that return various models based on needs
## It currently initializes MobileNetV2 and DenseNet with ImageNet weights, 
## and modifies the nodes to ensure they can run in the TPU environment
## (necessary during ~Summer 2019).

## TPUs require the batch_size to be set prior to running
###################################################################################
import sys
import tensorflow as tf


'''
This function initializes a MobileNetV2 model with ImageNet weights, then redefines 
the graph to include an explicitly-defined batch size to facilitate its deployment 
on TPUs. 

@param batch_size : The target batch_size
@param num_categories : Number of distinct classes requiring evaluation
@param wideslice : Boolean for whether or not a wideslice branch should be included.
For most use-cases, this is unnecessary and can probably be removed
@return a tf.keras.Model instance corresponding to MobileNetV2
'''
def MobNetVTwo(batch_size=None, num_categories=None, wideslice=False):
	model_input = tf.keras.Input(
		shape=(224,224,3),
		batch_size=batch_size, # batch size needs to be defined for TPU
		name='mobnetv2_in',
		dtype=tf.float32
	)

	base_model = tf.keras.applications.MobileNetV2(
		input_shape=(224,224,3),
		alpha=1.4,
		include_top=False,
		weights='imagenet',
		pooling='avg' # We include pooling since we do not include top
	)

	# This entire portion re-assigns nodes such that they statically-define batch size, 
	# which is necessary for the TPU

	# Removes input layer to assign a new one with batch_size defined
	base_model._layers.pop(0)

	temp = None 	# for facilitating properly calling "add"/merge layers
	block = 0	# increases to one on the first _project_BN layer
	model_output = None

	# The inbound nodes must be fixed for EACH layer, meaning we have to iterate over each and reset its value. The model must then be reformed properly
	for i in range(len(base_model._layers)):
		layer = base_model.get_layer(index=i)
		layer._inbound_nodes = []	# reset inbound nodes to remove the non-statically-defined batch size
		layer.trainable = False 	# dont allow training of pretrained weights

		if i == 0:
			model_output = layer(model_input)	# first non-input layer should be called with the model input
			continue

		if str(layer.name)[-4:] == '_add':
			model_output = layer([temp, model_output])	# merge layer requires two inputs

			add_block = block - 1	# add layers usually follow _project_BN, which increments block, but are in the same block. -1 accommodates this
			if add_block == 4 or add_block == 7 or add_block == 8 or add_block == 11 or add_block == 14:
				temp = model_output

		else:
			model_output = layer(model_output)	# all layers except merge should be called with the previous layer's output

			if str(layer.name)[-11:] == '_project_BN':
				if block == 1 or block == 3 or block == 6 or block == 10 or block == 13:
					temp = model_output 	# stores first project_bn layer into temp for merging

				block += 1	# increment block size

	# Incorporate wideslice branch if wideslice == True. This snippet was for my own experiments, so you can just delete it
	if (wideslice):
		wideslice_conv = tf.keras.layers.Conv2D(
			filters=160,
			kernel_size=(5, 224),
			kernel_initializer=tf.keras.initializers.he_normal(seed=None),
			name='wideslc_conv'
		)
		wideslice_conv.trainable = True
		wideslice_op = wideslice_conv(model_input)

		wideslice_bn = tf.keras.layers.BatchNormalization(
			axis=3, 
			epsilon=1.001e-5,
			name='wideslc_bn'
		)
		wideslice_op = wideslice_bn(wideslice_op)

		wideslice_relu = tf.keras.layers.Activation('relu', name='wideslc_relu')
		wideslice_op = wideslice_relu(wideslice_op)

		wideslice_maxpool = tf.keras.layers.MaxPool2D(
			pool_size=(5, 1),
			name='wideslc_mxpool'
		)
		wideslice_op = wideslice_maxpool(wideslice_op)

		wideslice_avgpool = tf.keras.layers.AveragePooling2D(
			pool_size=(22, 1),
			name='wideslc_avgpool'
		)
		wideslice_op = wideslice_avgpool(wideslice_op)

		#wideslice_op = tf.keras.layers.GlobalAveragePooling2D()(wideslice_op)
		wideslice_op = tf.keras.layers.Flatten(name='wideslc_flatten')(wideslice_op)

		concat_layer = tf.keras.layers.Concatenate(name='wideslc_cnct')
		model_output = concat_layer([wideslice_op, model_output])

	prediction_layer = tf.keras.layers.Dense(
		num_categories, 
		activation='softmax',
		kernel_initializer=tf.keras.initializers.he_normal(seed=None),
		name='final_dense'
	)
	prediction_layer.trainable = True	# this layer should be the only one trained
	predictions = prediction_layer(model_output)	# adds a softmax layer at the end

	return tf.keras.Model(inputs=[model_input], outputs=[predictions])


'''
This function initializes a DenseNet model with ImageNet weights, then redefines 
the graph to include an explicitly-defined batch size to facilitate its deployment 
on TPUs. 

@param batch_size : The target batch_size
@param num_categories : Number of distinct classes requiring evaluation
@param wideslice : Boolean for whether or not a wideslice branch should be included.
For most use-cases, this is unnecessary and can probably be removed
@return a tf.keras.Model instance corresponding to DenseNet
'''
def DenseNet(batch_size=None, num_categories=None, wideslice=False):
	model_input = tf.keras.Input(
		shape=(224,224,3),
		batch_size=batch_size, # batch size needs to be defined for TPU
		name='dense_in',
		dtype=tf.float32
	)
	base_model = tf.keras.applications.DenseNet121(
		input_shape=(224,224,3),
		include_top=False,
		weights='imagenet',
		pooling='avg' # We include pooling since we do not include top
	)

	# This entire portion re-assigns nodes such that they statically-define batch size, 
	# which is necessary for the TPU

	# Removes input layer to assign a new one with batch_size defined
	base_model._layers.pop(0)

	temp = None 	# for facilitating properly calling "add"/merge layers
	model_output = None

	# The inbound nodes must be fixed for EACH layer, meaning we have to iterate over each and reset its value. The model must then be reformed properly
	for i in range(len(base_model._layers)):
		layer = base_model.get_layer(index=i)
		layer._inbound_nodes = []	# reset inbound nodes to remove the non-statically-defined batch size
		layer.trainable = False 	# dont allow training of pretrained weights

		if i == 0:
			model_output = layer(model_input)	# first non-input layer should be called with the model input
			continue

		if str(layer.name)[-7:] == '_concat':
			model_output = layer([temp, model_output])	# merge layer requires two inputs
			temp = model_output

		else:
			model_output = layer(model_output)	# all layers except merge should be called with the previous layer's output

			if str(layer.name)[-5:] == '_pool' or str(layer.name) == 'pool1':
				temp = model_output		# for concatenation layers

	# Incorporate wideslice branch if wideslice == True. This snippet was for my own experiments, so you can just delete it
	if (wideslice):
		wideslice_conv = tf.keras.layers.Conv2D(
			filters=160,
			kernel_size=(5, 224),
			kernel_initializer=tf.keras.initializers.he_normal(seed=None),
			name='wideslc_conv'
		)
		wideslice_conv.trainable = True
		wideslice_op = wideslice_conv(model_input)

		wideslice_bn = tf.keras.layers.BatchNormalization(
			axis=3, 
			epsilon=1.001e-5,
			name='wideslc_bn'
		)
		wideslice_op = wideslice_bn(wideslice_op)

		wideslice_relu = tf.keras.layers.Activation('relu', name='wideslc_relu')
		wideslice_op = wideslice_relu(wideslice_op)

		wideslice_maxpool = tf.keras.layers.MaxPool2D(
			pool_size=(5, 1),
			name='wideslc_mxpool'
		)
		wideslice_op = wideslice_maxpool(wideslice_op)

		wideslice_avgpool = tf.keras.layers.AveragePooling2D(
			pool_size=(22, 1),
			name='wideslc_avgpool'
		)
		wideslice_op = wideslice_avgpool(wideslice_op)

		#wideslice_op = tf.keras.layers.GlobalAveragePooling2D()(wideslice_op)
		wideslice_op = tf.keras.layers.Flatten(name='wideslc_flatten')(wideslice_op)

		concat_layer = tf.keras.layers.Concatenate(name='wideslc_cnct')
		model_output = concat_layer([wideslice_op, model_output])

	prediction_layer = tf.keras.layers.Dense(
		num_categories, 
		activation='softmax',
		kernel_initializer=tf.keras.initializers.he_normal(seed=None),
		name='final_dense'
	)
	prediction_layer.trainable = True	# this layer should be the only one trained
	predictions = prediction_layer(model_output)	# adds a softmax layer at the end

	return tf.keras.Model(inputs=[model_input], outputs=[predictions])


'''
Experimental model to evaluate using DenseNet for feature extraction to train an MLP. Results
were not great, but this is left in for transparency. You can probably ignore this
'''
def DenseNetExtract(batch_size=None, num_categories=None, wideslice=False):
	model_input = tf.keras.Input(
		shape=(224,224,3),
		batch_size=batch_size,
		name='dense_in',
		dtype=tf.float32
	)
	base_model = tf.keras.applications.DenseNet121(
		input_shape=(224,224,3),
		include_top=False,
		weights='imagenet',
		pooling='avg' # Pooling included with model base as the global average pooling in the original model gets omitted without it when not including top
	)

	base_model._layers.pop(0)	# remove old input layer causing problems due to non-statically-defined batch size

	temp = None 	# for facilitating properly calling "add"/merge layers
	model_output = None

	# The inbound nodes must be fixed for EACH layer, meaning we have to iterate over each and reset its value. The model must then be reformed properly
	for i in range(len(base_model._layers)):
		layer = base_model.get_layer(index=i)
		layer._inbound_nodes = []	# reset inbound nodes to remove the non-statically-defined batch size
		layer.trainable = False 	# dont allow training of pretrained weights

		if i == 0:
			model_output = layer(model_input)	# first non-input layer should be called with the model input
			continue

		if str(layer.name)[-7:] == '_concat':
			model_output = layer([temp, model_output])	# merge layer requires two inputs
			temp = model_output

		else:
			model_output = layer(model_output)	# all layers except merge should be called with the previous layer's output

			if str(layer.name)[-5:] == '_pool' or str(layer.name) == 'pool1':
				temp = model_output		# for concatenation layers

	fc_layer = tf.keras.layers.Dense(
		2048, 
		activation='relu',
		kernel_initializer=tf.keras.initializers.he_normal(seed=None),
		name='secondlast_dense'
	)
	fc_layer.trainable = True
	model_output = fc_layer(model_output)

	dropout = tf.keras.layers.Dropout(0.5)	# combat overfitting
	model_output = dropout(model_output)

	prediction_layer = tf.keras.layers.Dense(
		num_categories, 
		activation='softmax',
		kernel_initializer=tf.keras.initializers.he_normal(seed=None),
		name='final_dense'
	)
	prediction_layer.trainable = True	# this layer should be the only one trained
	predictions = prediction_layer(model_output)	# adds a softmax layer at the end

	return tf.keras.Model(inputs=[model_input], outputs=[predictions])