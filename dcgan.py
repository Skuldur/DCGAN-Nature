import glob
import argparse
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Reshape
from keras.layers.convolutional import Convolution2D as Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from scipy import misc

from PIL import Image

def train(batch_size, learning_rate, beta_1, epochs, data_path):
    input_data = get_training_data(data_path)

    # normalize data between (-1, 1) which is the same output scale as tanh
    input_data = (input_data.astype(numpy.float32) - 127.5) / 127.5

    generator = get_generator()
    discriminator = get_discriminator()
    generative_adversarial_network = get_generative_adversarial_network(generator, discriminator)

    generator_optimizer = Adam(lr=learning_rate, beta_1=beta_1)
    discriminator_optimizer = Adam(lr=learning_rate, beta_1=beta_1)
    generator.compile(loss='binary_crossentropy', optimizer=generator_optimizer)
    generative_adversarial_network.compile(loss='binary_crossentropy', optimizer=generator_optimizer)
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=discriminator_optimizer)

    for epoch in range(epochs):
        for batch_number in range(int(input_data.shape[0]/batch_size)):
            input_batch = input_data[batch_number*batch_size: (batch_number+1)*batch_size]

            noise = numpy.random.uniform(-1, 1, size=(batch_size, 100))
            generated_images = generator.predict(noise, verbose=0)

            input_batch = numpy.concatenate((input_batch, generated_images))

            output_batch = [1] * batch_size + [0] * batch_size

            # train the discriminator to reject the generated images
            discriminator_loss = discriminator.train_on_batch(input_batch, output_batch)

            noise = numpy.random.uniform(-1, 1, (batch_size, 100))

            # we disable training the discriminator when training the generator since the
            # discriminator is being used to judge, we don't want to train it on false data
            discriminator.trainable = False

            # train the generator with the objective of getting the generated images approved
            generator_loss = generative_adversarial_network.train_on_batch(noise, [1] * batch_size)
            discriminator.trainable = True

            print("batch %d d_loss : %f" % (batch_number, discriminator_loss))
            print("batch %d g_loss : %f" % (batch_number, generator_loss))

        if epoch % 10 == 9:
            generator.save_weights('generator_weights', True)
            discriminator.save_weights('discriminator_weights', True)

def get_training_data(data_path):
    """ """
    input_data = []

    for image_path in glob.glob(data_path + '/*'):
        image = misc.imread(image_path)
        input_data.append(image)

    return numpy.array(input_data)

def get_generator():
    """ Create a model that takes in a matrix of random values as input and outputs images """
    model = Sequential()
    model.add(Dense(input_dim=100, units=1024))
    model.add(Activation('tanh'))
    model.add(Dense(128*8*8))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((8, 8, 128), input_shape=(128*8*8,))) # 8x8 image
    model.add(UpSampling2D(size=(2, 2))) # 16x16 image
    model.add(Conv2D(128, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2))) # 32x32 image
    model.add(Conv2D(128, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2))) # 64x64 image
    model.add(Conv2D(3, (5, 5), padding='same'))
    model.add(Activation('tanh'))

    return model

def get_discriminator():
    """ Create a model that takes in an image and outputs whether it contains our desired subject"""
    model = Sequential()
    model.add(
        Conv2D(
            64,
            (5, 5),
            padding='same',
            input_shape=(64, 64, 3)
            )
        )
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model

def get_generative_adversarial_network(generator, discriminator):
    """
    A network composed of a generator and discriminator network

    The flow of data is as follows:
        Input -> Generator -> Discriminator -> Output
    """
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model

def generate(batch_size, model_weights):
    """ Generate images using the trained generator """
    generator = get_generator()
    generator.compile(loss='binary_crossentropy', optimizer="Adam")
    generator.load_weights(model_weights)
    
    noise = numpy.random.uniform(-1, 1, (batch_size, 100))
    generated_images = generator.predict(noise, verbose=1)

    for i in range(batch_size):
        image = generated_images[i]
        image = image*127.5+127.5
        Image.fromarray(image.astype(numpy.uint8)).save(
        "generated_image-%s.png" % i)

# The default values in the arguments are the values recommended by the authors of
# the DCGAN paper (https://arxiv.org/pdf/1511.06434.pdf)
def get_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--learning_rate', default=0.0002, type=float)
    parser.add_argument('--beta_1', default=0.5, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--data_path', default='', type=str)
    parser.add_argument('--model', default='mountains', type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.mode == 'train':
        train(args.batch_size, args.learning_rate, args.beta_1, args.epochs, args.data_path)
    elif args.mode == 'generate':
        generate(args.batch_size, args.model + "_generator")