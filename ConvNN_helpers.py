"""
This file contains helper functions for the class 'JeongCnnTrainer', which is an inner class of the class 'JeongC2'
"""
import tensorflow as tf
import numpy as np
from random import shuffle, randint
import cv2 as cv
import os
import math as m

weights_init_hypp = 10


def conv_layer(namescope, inputs, nbr_of_filt, filt_h, filt_w, stride=1, padding='SAME', init_h_par=weights_init_hypp,
               dtype=tf.float16):
    """
    Apply convolutions and the ReLU activation on hl_alg_input data.
    :param init_h_par: a hyper-parameter to control the variance of a given kernel's weights.
    :return:
    """
    with tf.compat.v1.variable_scope(name_or_scope=namescope, reuse=tf.compat.v1.AUTO_REUSE) as varscope:
        # fil_d is the number of channels of the hl_alg_input. hl_alg_input usually has a shape = [batch_size,
        # height, width, # channels] filters' depth is the same as that on the hl_alg_input
        fil_d = inputs.shape[-1]  # inputs' depth is the last element of its shape
        # get the number of weights per one filter
        weights_nbr_per_filt = filt_h * filt_w * fil_d
        # standard deviation of the normal distribution from which we sample the weights.
        init_stdd = m.sqrt(init_h_par / weights_nbr_per_filt)

        filts_weights = tf.compat.v1.get_variable(name="conv_weights",
                                                  shape=[filt_h, filt_w, fil_d, nbr_of_filt],
                                                  initializer=tf.compat.v1.truncated_normal_initializer(
                                                      stddev=init_stdd))
        biases = tf.compat.v1.get_variable("conv_biases",
                                           shape=[nbr_of_filt],
                                           initializer=tf.compat.v1.constant_initializer(0))

        conv = tf.nn.conv2d(input=inputs,
                            filters=filts_weights,
                            strides=[1, stride, stride, 1],
                            padding=padding)
    return tf.nn.relu(conv + biases)


def pool_layer(inputs, filt_h, filt_w, stride=2, padding='VALID'):
    pool = tf.nn.max_pool(input=inputs,
                          ksize=[1, filt_h, filt_w, 1],
                          strides=[1, stride, stride, 1],
                          padding=padding)
    return pool


def fc_layer(namescope, inputs, nodes_nbr, weights_per_node_nbr, init_h_par=weights_init_hypp):
    """
    :param dtype:
    :param namescope:
    :param inputs:
    :param nodes_nbr:
    :param weights_per_node_nbr:
    :param init_h_par: a hyper-parameter to control the variance of a given kernel's weights.
    :return:
    """

    with tf.compat.v1.variable_scope(name_or_scope=namescope, reuse=tf.compat.v1.AUTO_REUSE) as varscope:
        total_weights_nbr = nodes_nbr * weights_per_node_nbr
        # standard deviation of the normal distribution from which we sample the weights.
        init_stdd = m.sqrt(init_h_par / total_weights_nbr)
        # weights parameters of all nodes of this fc layer
        nodes_weights = tf.compat.v1.get_variable(name="fc_weights",
                                                  shape=[weights_per_node_nbr, nodes_nbr],
                                                  initializer=tf.compat.v1.truncated_normal_initializer(
                                                      stddev=init_stdd))
        # biases parameters of all nodes of this fc layer
        biases = tf.compat.v1.get_variable(name="fc_biases",
                                           shape=[nodes_nbr],
                                           initializer=tf.compat.v1.constant_initializer(0))
        return tf.nn.relu(tf.matmul(inputs, nodes_weights) + biases)


def out_layer(namescope, inputs, weights_per_node_nbr, init_h_par=weights_init_hypp):
    with tf.compat.v1.variable_scope(name_or_scope=namescope, reuse=tf.compat.v1.AUTO_REUSE) as varscope:
        total_weights_nbr = weights_per_node_nbr
        init_stdd = m.sqrt(init_h_par / total_weights_nbr)
        # weights parameters of all nodes of this fc layer
        nodes_weights = tf.compat.v1.get_variable(name="fc_weights",
                                                  shape=[weights_per_node_nbr, 1],
                                                  initializer=tf.compat.v1.truncated_normal_initializer(
                                                      stddev=init_stdd))
        # biases parameters of all nodes of this fc layer
        biases = tf.compat.v1.get_variable(name="fc_biases",
                                           shape=[1],
                                           initializer=tf.compat.v1.constant_initializer(0))

        # NOTE: ####################################################################
        # NO ACTIVATION FUNCTION IS APPLIED IN THE FINAL LAYER.
        # SUCH ACTIVATION IS APPLIED INSIDE THE LOSS FUNCTION FOR EFFICIENCY REASONS.
        ############################################################################

        return tf.matmul(inputs, nodes_weights) + biases


def create_batch_as_np(data_path, shuffle_data=False):
    """
    This function returns two batches; one for samples and the other for corresponding binary labels, respectively.
    The term samples in this function refers to image samples. :param data_path: path that must contain only two
    folders: one folder contains samples of the first class and the other folder contains samples of the second
    class. :return: samples_batch, a 4d numpy.array of shape (number_of_samples, image_height, image_width,
    number_of_channels) labels_batch, a 2d numpy.array of shape (number_of_samples, 1). The second dimension is 1
    because only a scalar is needed for binary labels (0 for class 0, 1 for class 1).
    """
    # create data_list with this structure: [[sample1,label1],[sample2,label2],...,[sampleN,labelN]]
    data_list = []
    # loop over folder of class 1
    # for each iteration
    # read one sample and put it in a list sample_list = [sample, class1];
    # append sample_list to data_list
    # path of data, which must contain folders, each of which containing samples of a unique class.

    classes_dir = os.listdir(data_path)  # get a list of names of folders containing training samples.

    label = 0  # this variable encodes labels. When the program loops over the first class, label = 0. When looping
    # over the next class, label get incremented by 1.
    for class_dir in classes_dir:  # iterates as many as the number of classes (i.e., 2 classes, 2 iterations)
        class_path = os.path.join(data_path, class_dir)  # create the path to a class folder
        samples_filenames = os.listdir(class_path)  # filenames of all class samples
        for sample_filename in samples_filenames:  # read all class samples stored in class_path
            sample_path = os.path.join(class_path, sample_filename)  # create the sample path
            sample = cv.imread(sample_path)  # read the sample/image
            data_list.append([sample, label])
        # after iterating over all samples of a given class, increment label
        label += 1

    # Shuffle the data
    if shuffle_data:
        shuffle(data_list)  # I think this allows gradient descent to work better

    # get a list of samples and a list of labels by looping over data_list
    samples_list = []
    labels_list = []
    for sample, label in data_list:
        samples_list.append(sample)
        labels_list.append(np.array([label]))

    # Stack the list of samples into a 4D nd array called X.
    samples_batch = np.stack(samples_list)  # samples_batch.shape = (#_of_samples, img_height, img_width, #channels)
    # Stack the list of labels into a 2D array called Y
    labels_batch = np.stack(
        labels_list)  # labels_batch.shape = (#_of_samples, 1); 1 because we deal binary classification

    # Normalization: map image pixels from [0, 255] to [0, 1] (dtype = np.float64)
    # samples_batch = np.divide(samples_batch, 255)

    # dtype of samples_batch and labels_batch must be the same. Otherwise, a Type error raises when computing the loss
    # using the command tf.nn.sigmoid_cross_entropy_with_logits
    labels_batch = labels_batch.astype(dtype=samples_batch.dtype)
    return samples_batch, labels_batch


def augment_batch(batch):
    """
    Augments the hl_alg_input batch (of images) using horizontal flip.
    :param batch: a tuple
    :return: augmented batch as a tupe
    """
    samples_batch, labels_batch = batch
    del batch
    flipped_samples_batch = tf.image.flip_left_right(image=samples_batch)

    # Convert flipped_samples_batch into numpy
    if tf.executing_eagerly():  # True if eager mode is enabled
        # eager state = Enabled
        # print("eager state = Enabled")
        flipped_samples_batch = flipped_samples_batch.numpy()
    else:
        # eager state = Disabled
        # print("eager state = Disabled")
        with tf.compat.v1.Session() as sess:
            flipped_samples_batch = sess.run(flipped_samples_batch)

    aug_samples_batch = np.append(arr=samples_batch, values=flipped_samples_batch, axis=0)
    aug_labels_batch = np.append(arr=labels_batch, values=labels_batch, axis=0)  # same array is appended because
    # data augmentation does not change the label.
    return aug_samples_batch, aug_labels_batch


def normalize_batch(batch):
    """
    Normalizes samples/images by scaling pixel values to the range 0-1.
    :param batch: either a tuple of samples_batch, labels_batch, or samples_batch only.
    :return: normalized samples_batch
    """
    if type(batch).__name__ == 'tuple':
        norm_samples_batch, labels_batch = batch
        del batch
        # Normalization: map image pixels from [0, 255] to [0, 1] (dtype = np.float64)
        norm_samples_batch = np.divide(norm_samples_batch, 255)
        # dtype of samples_batch and labels_batch must be the same. Otherwise, a Type error raises when computing the
        # loss using the command tf.nn.sigmoid_cross_entropy_with_logits
        norm_samples_batch = norm_samples_batch.astype(dtype=np.float16)
        labels_batch = labels_batch.astype(dtype=norm_samples_batch.dtype)
        return norm_samples_batch, labels_batch
    else:
        norm_samples_batch = np.divide(batch, 255)
        del batch
        norm_samples_batch = norm_samples_batch.astype(dtype=np.float16)
        return norm_samples_batch


def visualize_dataset(batch):
    """
    Enables the user to visualze dataset samples with corresponding labels. I needed this method to make sure that
    my dataset is correctly labeled.

    How it works: ------------- When using this method, sample images are shown randomly in a full-screen window. The
    shown sample is overlayed with a text formatted as follows: "label sample_order"; where label is either: H (for
    horizon) or NH (for Non- horizon) and sample_order is the order of the sample within its samples_batch. For
    instance, NH 99^th means the shown image is the 99^th sample labeled Non-horizon.

    :param batch: a tuple of samples_batch samples and samples_batch labels.
    :return: None
    """
    batch_samples, batch_labels = batch
    samples_nbr = batch_samples.shape[0]
    index = 0
    while True:
        label = batch_labels[index][0]
        if label == 1:
            text = "NH"
            color = (0, 0, 1)
        elif label == 0:
            text = "H"
            color = (0, 1, 0)
        else:
            text = "Unexpected error"
            color = (0, 0, 0)
        title = "Patch and Label"
        img = batch_samples[index]
        img = np.float64(img)
        img = cv.resize(src=img, dsize=(150, 150))  # resize is important to clarify put text
        text = text + " " + str(index) + "^th"
        img = cv.putText(img=img, text=text, org=(0, 13), fontFace=0, fontScale=0.5, color=color)
        cv.namedWindow(winname=title, flags=cv.WINDOW_NORMAL)
        cv.setWindowProperty(winname=title, prop_id=cv.WND_PROP_FULLSCREEN, prop_value=cv.WINDOW_FULLSCREEN)
        cv.imshow(title, img)
        pressed_key = cv.waitKey()
        if pressed_key == ord('q'):
            cv.destroyAllWindows()
            break
        else:
            index = randint(a=0, b=samples_nbr - 1)
            print(index)
        cv.destroyAllWindows()


def model_accuracy(predictions, labels):
    # positive and values of the hl_alg_input 'prediction_batches' correspond to class 1 and class 0, respectively,
    # because a sigmoid function of positive values is > 0.5 (i.e., 50% probability of being class 1).
    predictions = np.where(predictions > 0, 1, 0)  # if the condition :predictions > 0 is true replace element by 1.
    # Otherwise, replace by 0. predictions_batch is a 2d array of shape (#_of_samples,1), containing either the value
    # 0 or 1, which encode the class 0 and class 1. correct is a 2D array of shape (#_of_samples,1), containing
    # either 1 or 0. the value 1 means that the sample is correctly classified and 0 means otherwise.
    correct = np.where(predictions == labels, 1, 0)
    # the accuracy is the mean of the array correct, multiplied by 100 (to get percentage value). For example,
    # if the array correct contains 3 values = 1 and 7 values = 0, then the accuracy  = mean = (0*7 + 3*1)/10 =
    # 0.3*100 = 30%
    return np.mean(correct) * 100


def shuffle_folder_images(src_dir, dst_dir):
    """
    Shuffles images from src_dir and stores them in dst_dir
    :param src_dir: source f_dir of images to shuffle
    :param dst_dir: destination f_dir of shuffled images
    :return: nothing

    Code example
    """
    list_of_dirs = os.listdir(src_dir)
    shuffle(list_of_dirs)
    new_order_list = list(range(0, len(list_of_dirs)))
    for new_order in new_order_list:
        image = cv.imread(os.path.join(src_dir, list_of_dirs[new_order]))
        dst_file_path = os.path.join(dst_dir, str(new_order) + ".png")
        cv.imwrite(filename=dst_file_path, img=image)
