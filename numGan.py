# necessary imports
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from IPython import display

# load MNIST Dataset
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

# resize the images
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]

# amount of data in each batch
BUFFER_SIZE = 60000
BATCH_SIZE = 256

# batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# create a generator that uses tf.keras.layers.Conv2DTranspose (upsampling) layers to produce an image from a seed (random noise)
def make_generator_model():
	model = tf.keras.Sequential()
	model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())

	model.add(layers.Reshape((7, 7, 256)))
	assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

	model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
	assert model.output_shape == (None, 7, 7, 128)
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())

	model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
	assert model.output_shape == (None, 14, 14, 64)
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())

	model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
	assert model.output_shape == (None, 28, 28, 1)

	return model

# create an image with the untrained generator
generator = make_generator_model()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')

# create the discriminaator
def make_discriminator_model():
	model = tf.keras.Sequential()
	model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
	model.add(layers.LeakyReLU())
	model.add(layers.Dropout(0.3))

	model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
	model.add(layers.LeakyReLU())
	model.add(layers.Dropout(0.3))

	model.add(layers.Flatten())
	model.add(layers.Dense(1))

	return model

# use untrained discriminator to classify the generated images as real or fake
discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print (decision)

# define loss function (I am using entropy loss)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# this method calculates discriminator loss - distinguising real images from fake
def discriminator_loss(real_output, fake_output):
	# find loss from real image (1's)
	real_loss = cross_entropy(tf.ones_like(real_output), real_output)
	# find loss from fake image (0's)
	fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
	# total loss is sum of these two losses
	total_loss = real_loss + fake_loss
	return total_loss

# generator tries to trick the discriminator into thinking its the real output 
def generator_loss(fake_output):
	# so its loss will compare the discriminators decisions on the generated images to an array of 1s
	return cross_entropy(tf.ones_like(fake_output), fake_output)

# each network is separately trained, so their optimizers are independent
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# save checkpoints for long term training (can restore models from checkpoints)
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer, generator=generator, discriminator=discriminator)

# defining the training loop
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

# We will reuse this seed overtime (so it's easier) to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# the annotation `tf.function` causes the function to be "compiled".
@tf.function
def train_step(images):
	# get random seed as input
	noise = tf.random.normal([BATCH_SIZE, noise_dim])


	with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
		generated_images = generator(noise, training=True)

		# discriminator is used to classify real images (from training set) and fake images (from generator), then calculate the loss in order to update them
		real_output = discriminator(images, training=True)
		fake_output = discriminator(generated_images, training=True)

		gen_loss = generator_loss(fake_output)
		disc_loss = discriminator_loss(real_output, fake_output)


	gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
	gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

	# generator and discriminator are then updated (optimized)
	generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
	discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# call previous train step function during training
def train(dataset, epochs):
	for epoch in range(epochs):
		start = time.time()

		for image_batch in dataset:
			train_step(image_batch)

		# produce images for the GIF as we go
		display.clear_output(wait=True)
		generate_and_save_images(generator, epoch + 1, seed)

		# save the model every 15 epochs
		if (epoch + 1) % 15 == 0:
			checkpoint.save(file_prefix = checkpoint_prefix)

		print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

	# generate after the final epoch
	display.clear_output(wait=True)
	generate_and_save_images(generator, epochs, seed)

# method to generate and save images based off inputted seed
def generate_and_save_images(model, epoch, test_input):
	# notice `training` is set to False so all layers run in inference mode (batchnorm)
	predictions = model(test_input, training=False)

	fig = plt.figure(figsize=(4,4))

	for i in range(predictions.shape[0]):
		plt.subplot(4, 4, i+1)
		plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
		plt.axis('off')

	plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
	# plt.show() if you want to show image at each epoch

# call train to train both the discriminator and generator independently
train(train_dataset, EPOCHS)

# restore the latest checkpoint
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# display an image at an epoch number
def display_image(epoch_no):
	return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

# display the image at the epoch
display_image(EPOCHS)

# use imageio to create an animated gif using the images saved during training
anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
	filenames = glob.glob('image*.png')
	filenames = sorted(filenames)
	last = -1
	for i,filename in enumerate(filenames):
		frame = 2*(i**0.5)
		if round(frame) > round(last):
			last = frame
		else:
			continue
		image = imageio.imread(filename)
		writer.append_data(image)
	image = imageio.imread(filename)
	writer.append_data(image)

import IPython
if IPython.version_info > (6,2,0,''):
	display.Image(filename=anim_file)


