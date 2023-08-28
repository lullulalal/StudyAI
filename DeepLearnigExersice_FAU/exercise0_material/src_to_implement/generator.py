import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import math
import skimage.transform
import random

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        #TODO: implement constructor

        # Load labels in json as dictionary
        with open(label_path, 'r') as f:
            self.label_dict = json.load(f)

        # Load all images to dictionary
        self.image_dict = {}
        for npy_name in self.label_dict.keys():
            self.image_dict[npy_name] = np.load(file_path + npy_name + '.npy')

        # Get key list from label_dict
        # key_list is used to access the label_dict and image_dict
        # In next() method, access the label_dict and image_dict in the order of key_list
        # So, We do not shuffle label_dict and image_dict directly.
        self.key_list = list(self.label_dict.keys())

        # Calculate reusing number of images in last batch
        # The length of key_list is total number of images
        self.reusing_num = 0
        remainder = len(self.key_list) % batch_size
        if remainder != 0: self.reusing_num = batch_size - remainder

        # Other parameters save to instance variables
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle

        # Define other instance variables
        self.total_batch = math.ceil(len(self.key_list) / batch_size)
        self.current_batch = self.total_batch
        self.epoch_num = -1


    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        #TODO: implement next method

        # Check if one epoch is finished.
        # Since it was initialized as self.current_batch = self.total_batch in the constructor,
        # it must be run on the first call of next().
        if self.current_batch == self.total_batch:
            self.current_batch = 0
            self.epoch_num += 1
            if self.shuffle: random.shuffle(self.key_list)

        # Create batch
        # Get keys for the current batch.
        start = self.current_batch * self.batch_size
        end = start + self.batch_size
        keys = self.key_list[start:end]

        if self.current_batch == self.total_batch - 1:
            keys += self.key_list[0:self.reusing_num]

        index = 0
        # Define output buffer
        labels = np.empty(self.batch_size, dtype=int)
        images = np.empty(shape=[self.batch_size] + self.image_size)
        for key in keys:
            labels[index] = self.label_dict[key]
            image = skimage.transform.resize(self.image_dict[key], self.image_size)
            images[index] = self.augment(image)
            index += 1

        # Update current batch
        self.current_batch += 1

        return images, labels

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function

        if self.rotation:
            rot_num = np.random.randint(0, 4)
            img = np.rot90(img, rot_num)

        if self.mirroring:
            flag = np.random.randint(0, 2)
            if flag == 1: img = np.fliplr(img)

        return img

    def current_epoch(self):
        # return the current epoch number
        return self.epoch_num

    def class_name(self, x):
        # This function returns the class name for a specific input
        #TODO: implement class name function

        return self.class_dict[x]

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method

        images, labels = self.next()
        column_num = 3
        row_num = math.ceil(self.batch_size / column_num)

        fig = plt.figure()

        for index in range(self.batch_size):
            temp = fig.add_subplot(row_num, column_num, index+1)
            temp.imshow(images[index])
            temp.title.set_text(self.class_name(labels[index]))
            temp.axis('off')

        plt.show()