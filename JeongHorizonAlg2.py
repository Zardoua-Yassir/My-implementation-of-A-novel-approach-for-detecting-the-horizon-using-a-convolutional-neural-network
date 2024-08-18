"""
Implementation by Yassir Zardoua, 2020.
Contact: yassirzardoua@gmail.com
"""
import tensorflow as tf
import ConvNN_helpers as cnnh
import constants as cts
import numpy as np
import os
from math import isnan
from tkinter import messagebox
import cv2 as cv
from math import pi, atan
from time import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # useful to speed-up TF computations
tf.compat.v1.disable_eager_execution()


class CnnModelJeong:
    """
    A class to implement the horizon detection algorithm published in Jeong et al's paper.
    Paper link: https://link.springer.com/article/10.1007/s11045-018-0602-4
    """

    def __init__(self, model_pars_path):

        """
        This algorithm must have two attributes to work: a trained graph for the CNN.
        :param model_pars_path: path to the trained cnn model
        """
        self.testing_sess = tf.compat.v1.Session()
        self.trained_cnn = JeongTrainedCnn(trained_pars_path=model_pars_path, session=self.testing_sess)
        # create an instance of the candidate pixel extractor
        self.cand_pixs_collector = JeongPixs()
        # detecting the horizon can be done by calling the method self.get_horizon
        self.detected_hl_flag = True

    def get_horizon(self, img):
        """
        Processes the image img The processing aims to:
            1) find candidate edge pixels
            2) classify them using a trained CNN as a horizon or non-horizon
            3) fit the horizon line
        :return:
        """
        # get candidate edge pixels
        self.start_time = time()
        self.img = img
        self.cand_pixs_xy, _ = self.cand_pixs_collector.get_cand_pixs(src_im=self.img)
        self.cand_pixs_x, self.cand_pixs_y = self.cand_pixs_xy
        self.img_height, self.img_width, _ = self.img.shape
        # keep only in-frame candidate edge pixels
        self.in_frame_cand_pixs_xy = self.cand_pixs_collector.in_frame_xy(x=self.cand_pixs_x,
                                                                          y=self.cand_pixs_y,
                                                                          patch_w=32,
                                                                          patch_h=32,
                                                                          frame_w=self.img_width,
                                                                          frame_h=self.img_height)
        # extract patches centered at candidate edge pixels as a batch
        self.patches_batch = self._get_patches(src_im=self.img, cand_pixs_xy=self.in_frame_cand_pixs_xy)
        # normalize the extracted patches
        self.patches_batch = cnnh.normalize_batch(batch=self.patches_batch)  # normalize the samples_batch

        # classify the patches (0 corresponds to horizon pixels, 1 to non-horizon pixels)
        self.predicted_classes = self.trained_cnn.classify_patches(patches_batch=self.patches_batch)
        # draw classified pixels; get the image with drawn classified pixels
        self.get_classified_pixs()

        # self.rectified_edge_map = np.zeros((self.img_height, self.img_width), dtype=np.uint8)  # the edge map with
        # # pixels classified as horizon
        # self.rectified_edge_map[self.h_pixs_y, self.h_pixs_x] = 255
        # cv.imwrite("rectified_edge_map.png", self.rectified_edge_map)

        self._outlier_elimination_of_hl_pixs()

        # self.outlier_eliminated_edge_map = np.zeros(shape=(self.img_height, self.img_width), dtype=np.uint8)
        # self.outlier_eliminated_edge_map[self.h_pixs_y, self.h_pixs_x] = 255
        # cv.imwrite("outlier_eliminated_edge_map.png", self.outlier_eliminated_edge_map)

        self.end_time = time()
        self.latency = round((self.end_time - self.start_time), 4)

        if not self.detected_hl_flag:
            self.det_position_hl = np.nan
            self.det_tilt_hl = np.nan
            self.latency = np.nan
        self.detected_hl_flag = True

    def _outlier_elimination_of_hl_pixs(self):
        for fitting_iteration in range(3):  # the outlier elimination process (requires 3 iterations)
            self._fit_with_least_squares()  # fit a line on current horizon edge pixels
            # compute distance between pixels in self.h_pixs_xy and the fit line
            if not self.detected_hl_flag:
                return
            self.line_pixs_distances = np.divide(np.abs(
                np.add(np.subtract(np.multiply(self.hl_slope, self.h_pixs_x), self.h_pixs_y), self.hl_intercept)),
                np.sqrt(self.hl_slope ** 2 + 1))
            # compute the median distance of all distances
            self.med_distance = np.median(self.line_pixs_distances)
            # find indexes of pixels whose distance from fit line is smaller than the median. Get their indexes
            #  and update the value of self.h_pixs_xy
            self.h_pixs_xy_indexes = np.where(self.line_pixs_distances < self.med_distance)
            self.h_pixs_x = self.h_pixs_x[self.h_pixs_xy_indexes]
            self.h_pixs_y = self.h_pixs_y[self.h_pixs_xy_indexes]
            # repeat the fitting process two more times
            self._fit_with_least_squares()  # a final fit on resulted survived horizon pixels.
            self.det_tilt_hl = (-atan(self.hl_slope)) * (180 / pi)  # - because the y axis of images goes down
            self.det_position_hl = ((self.img_width - 1) / 2) * self.hl_slope + self.hl_intercept

    def _fit_with_least_squares(self):
        self.h_pixs_xy = np.zeros((self.h_pixs_x.size, 2), dtype=np.int32)
        self.h_pixs_xy[:, 0], self.h_pixs_xy[:, 1] = self.h_pixs_x, self.h_pixs_y
        self.check_detection_possility()
        if not self.detected_hl_flag:
            return

        [vx, vy, x, y] = cv.fitLine(points=self.h_pixs_xy, distType=cv.DIST_L2,
                                    param=0, reps=1, aeps=pi / 180)

        self.hl_slope = float(vy / vx)  # float to convert from (1,) float numpy array to python float
        self.hl_intercept = float(y - self.hl_slope * x)

    def draw_hl(self):
        """
        Draws the horizon line on attribute 'self.img_with_hl' if it is detected. Otherwise, the text 'NO HORIZON IS
        DETECTED' is put on the image.
        """
        self.img_with_hl = np.copy(self.img)
        if self.detected_hl_flag:
            self.xs_hl = int(0)
            self.xe_hl = int(self.img_width - 1)
            self.ys_hl = int(self.hl_intercept)  # = int((self.hl_slope * self.xs_hl) + self.hl_intercept)
            self.ye_hl = int((self.xe_hl * self.hl_slope) + self.hl_intercept)
            cv.line(self.img_with_hl, (self.xs_hl, self.ys_hl), (self.xe_hl, self.ye_hl), (0, 0, 255), 5)

    def check_detection_possility(self):
        """
        Checks if the horizon can still be detected.
        """
        if self.h_pixs_xy.size < 2:  # because line fitting requires at least two points
            self.detected_hl_flag = False

    def evaluate(self, src_video_folder, src_gt_folder, dst_video_folder=r"", dst_quantitative_results_folder=r"",
                 draw_and_save=True):
        """
        Produces a .npy file containing quantitative results of the Horizon Edge Filter algorithm. The .npy file
        contains the following information for each image: |Y_gt - Y_det|, |alpha_gt - alpha_det|, and latency in
        seconds
        between 0 and 1) specifying the ratio of the diameter of the resized image being processed. For instance, if
        the attributre self.dsize = (640, 480), the threshold that will be used in the hough transform is sqrt(640^2 +
        480^2) * hough_threshold_ratio, rounded to the nearest integer.
        :param src_gt_folder: absolute path to the ground truth horizons corresponding to source video files.
        :param src_video_folder: absolute path to folder containing source video files to process
        :param dst_video_folder: absolute path where video files with drawn horizon will be saved.
        :param dst_quantitative_results_folder: destination folder where quantitative results will be saved.
        :param draw_and_save: if True, all detected horizons will be drawn on their corresponding frames and saved as
        video files
        in the folder specified by 'dst_video_folder'.
        """
        src_video_names = sorted(os.listdir(src_video_folder))
        srt_gt_names = sorted(os.listdir(src_gt_folder))
        for src_video_name, src_gt_name in zip(src_video_names, srt_gt_names):
            print("{} will correspond to {}".format(src_video_name, src_gt_name))

        # Allowing the user to verify that each gt .npy file corresponds to the correct video file # # # # # # # # # # #
        while True:
            yn = input("Above are the video files and their corresponding gt files. If they are correct, click on 'y'"
                       " to proceed, otherwise, click on 'n'.\n"
                       "If one or more video file has incorrect gt file correspondence, we recommend to rename the"
                       "files with similar names.")
            if yn == 'y':
                break
            elif yn == 'n':
                print("\nTHE QUANTITATIVE EVALUATION IS ABORTED AS ONE OR MORE LOADED GT FILES DOES NOT CORRESPOND TO "
                      "THE CORRECT VIDEO FILE")
                return
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        self.det_horizons_all_files = np.empty(shape=[0, 5])
        for src_video_name, src_gt_name in zip(src_video_names, srt_gt_names):  # each iteration processes one video
            # file
            print("loaded video/loaded gt: {}/{}".format(src_video_name, src_gt_name))  # printing which video file
            # correspond to which gt file

            src_video_path = os.path.join(src_video_folder, src_video_name)
            src_gt_path = os.path.join(src_gt_folder, src_gt_name)

            cap = cv.VideoCapture(src_video_path)  # create a video reader object
            # Creating the video writer # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            fps = cap.get(propId=cv.CAP_PROP_FPS)
            self.img_width = int(cap.get(propId=cv.CAP_PROP_FRAME_WIDTH))
            self.img_height = int(cap.get(propId=cv.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')  # codec used to compress the video.
            if draw_and_save:
                dst_vid_path = os.path.join(dst_video_folder, "C.Jeo_2_" + src_video_name)
                video_writer = cv.VideoWriter(dst_vid_path, fourcc, fps, (self.img_width, self.img_height),
                                              True)  # video writer object
            self.gt_horizons = np.load(src_gt_path)
            #
            nbr_of_annotations = self.gt_horizons.shape[0]
            nbr_of_frames = int(cap.get(propId=cv.CAP_PROP_FRAME_COUNT))
            if nbr_of_frames != nbr_of_annotations:
                error_text_1 = "The number of annotations (={}) does not equal to the number of frames (={})". \
                    format(nbr_of_annotations, nbr_of_frames)
                raise Exception(error_text_1)

            self.det_horizons_per_file = np.zeros((nbr_of_annotations, 5))
            for idx, gt_horizon in enumerate(self.gt_horizons):
                no_error_flag, frame = cap.read()
                if not no_error_flag:
                    break
                self.get_horizon(img=frame)  # gets the horizon position and
                # tilt
                self.gt_position_hl, self.gt_tilt_hl = gt_horizon
                print("detected position/gt position {}/{};\n detected tilt/gt tilt {}/{}".
                      format(round(self.det_position_hl, 2), round(self.gt_position_hl, 2), round(self.det_tilt_hl, 2),
                             round(self.gt_tilt_hl, 2)))
                print("with latency = {} seconds".format(round(self.latency, 4)))
                self.det_horizons_per_file[idx] = [self.det_position_hl,
                                                   self.det_tilt_hl,
                                                   round(abs(self.det_position_hl - self.gt_position_hl), 4),
                                                   round(abs(self.det_tilt_hl - self.gt_tilt_hl), 4),
                                                   self.latency]
                try:
                    self.draw_hl()  # draws the horizon on self.img_with_hl
                except:
                    pass
                video_writer.write(self.img_with_hl)
            cap.release()
            video_writer.release()
            print("The video file {} has been processed.".format(src_video_name))

            # saving the .npy file of quantitative results of current video file # # # # # # # # # # # # # # # # # # # #
            src_video_name_no_ext = os.path.splitext(src_video_name)[0]
            det_horizons_per_file_dst_path = os.path.join(dst_quantitative_results_folder,
                                                          src_video_name_no_ext + ".npy")
            np.save(det_horizons_per_file_dst_path, self.det_horizons_per_file)
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            self.det_horizons_all_files = np.append(self.det_horizons_all_files,
                                                    self.det_horizons_per_file,
                                                    axis=0)
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # after processing all video files, save quantitative results as .npy file
        src_video_folder_name = os.path.basename(src_video_folder)
        dst_detected_path = os.path.join(dst_quantitative_results_folder,
                                         "Jeong2_all_det_hl_" + src_video_folder_name + ".npy")
        np.save(dst_detected_path, self.det_horizons_all_files)

    def draw_circles(self, img, pixs_xy, color=(0, 0, 255)):
        pixs_x, pixs_y = pixs_xy
        for x, y in zip(pixs_x, pixs_y):
            img = cv.circle(img=img, center=(x, y), radius=4, color=color, thickness=4)
        return img

    def draw_classified_pixs(self, img, cand_pixs, predictions, show=False):
        """
        :param img: :param cand_pixs: :param predictions: :param show: if True, a window containing the image after
        drawing classified pixels pops-up :return: the input image img after drawing horizon pixels with green
        circles and nn-horizon pixels with red circles
        """
        nbr_of_predictions = predictions.size
        predictions = predictions.reshape(nbr_of_predictions)

        pixs_x, pixs_y = cand_pixs
        h_pixs_xy_indx = np.where(predictions == 0)
        nh_pixs_xy_indx = np.where(predictions == 1)
        h_pixs_x = pixs_x[h_pixs_xy_indx]
        h_pixs_y = pixs_y[h_pixs_xy_indx]

        nh_pixs_x = pixs_x[nh_pixs_xy_indx]
        nh_pixs_y = pixs_y[nh_pixs_xy_indx]

        img = self.draw_circles(img=img, pixs_xy=(h_pixs_x, h_pixs_y), color=(0, 255, 0))
        img = self.draw_circles(img=img, pixs_xy=(nh_pixs_x, nh_pixs_y), color=(0, 0, 255))
        img = cv.resize(src=img, dst=img, dsize=(1300, 800))
        if show:
            title = 'classified pixels: horizon in green, non-horizon in red'
            cv.imshow(title, img)
            cv.waitKey()
            cv.destroyWindow(title)
        return img

    def _get_patches(self, src_im, cand_pixs_xy, patch_w=32, patch_h=32):
        """
        Extract patches at candidate edge pixels :param cand_pixs_xy: a tupe (cand_pixs_x,y), where cand_pixs_x and y
        contain cand_pixs_x and y coordinates of all candidate pixels :param patch_w: width of patches to extract
        :param patch_h: height of patches to extract :return: a samples_batch of patches of shape: (
        number_of_cand_pixs, 32, 32, number_of_img_channels)
        """
        half_patch_w = int(patch_w / 2)  # compute patch_width/2: useful to identify endpoints of patches
        half_patch_h = int(patch_h / 2)  # compute patch_height/2: useful to identify endpoints of patches
        pixs_x, pixs_y = cand_pixs_xy
        patches_nbr = pixs_x.size
        if patches_nbr != pixs_y.size:
            raise Exception(
                "the two elements of the tuple argument 'cand_pixs_xy', which are numpy arrays, do not seem to have "
                "the same size!")
        patches_batch = np.zeros((patches_nbr, 32, 32, 3), dtype=np.float16)
        patch_indx = 0
        for pix_x, pix_y in zip(pixs_x, pixs_y):
            xs = pix_x - half_patch_w
            xe = xs + patch_w
            ys = pix_y - half_patch_h
            ye = ys + patch_h
            patches_batch[patch_indx] = src_im[ys:ye, xs:xe]
            patch_indx += 1
        return patches_batch

    def save_img(self, dst_dir, img, prefix='img'):
        """
        Saves an image as a .png file in a directory. All saved images are dynamically suffixed by a number representing
        their order within the destination directory.
        :param dst_dir: the directory to save the image in
        :param img: the image to save
        :param prefix: the prefix of the image filename
        :return:
        """
        if os.path.exists(dst_dir):
            pass
        else:
            dst_dir = os.path.join(os.getcwd(), 'alg_output')
            if not os.path.exists(dst_dir):
                os.mkdir(dst_dir)
        suffix = str(len(os.listdir(dst_dir)))
        file_name = prefix + '_' + suffix + '.png'
        file_abs_path = os.path.join(dst_dir, file_name)
        cv.imwrite(filename=file_abs_path, img=img)

    def listdirs(self, path, ext: str, abs_path=False):
        """
        Returns a list of directories with a specific extension inside a given folder
        :param path: the path of the folder to list the directories from
        :param ext: the extension of the directories to return
        :param abs_path: if True, returned directories are absolute paths
        :return: a list of directories with a specific extension
        """
        all_files = os.listdir(path)
        files_with_ext = []
        for file in all_files:
            filename, extension = os.path.splitext(file)
            if extension == ext:
                if abs_path:
                    file = os.path.join(path, file)  # this makes file as an absolute path
                files_with_ext.append(file)
        return files_with_ext

    def get_classified_pixs(self):
        """
        uses results of patches classification, stored in self.predicted_classes, to get (cand_pixs_x,y) coordinates
        of pixels classified as horizon, self.h_pixs_x and self.h_pixs_y, and pixels classified as non-horizon,
        self.nh_pixs_x and self.nh_pixs_y.
        """
        nbr_of_predictions = self.predicted_classes.size
        predictions = self.predicted_classes.reshape(nbr_of_predictions)

        pixs_x, pixs_y = self.cand_pixs_xy
        h_pixs_xy_indx = np.where(predictions == 0)
        nh_pixs_xy_indx = np.where(predictions == 1)
        self.h_pixs_x = pixs_x[h_pixs_xy_indx]
        self.h_pixs_y = pixs_y[h_pixs_xy_indx]


class CnnTrainer:
    """
    A class to train the Convolutional Neural Network model of Jeong et al's, published in paper whose link is:
    https://link.springer.com/article/10.1007/s11045-018-0602-4

    :param train_dataset_path: dataset to train the algorithm's classifier (must be provided in trained_alg=False).
    :param model_pars_path: path to the trained parameters of the model to load :param test_dataset_path: dataset to
    test the algorithm's classifier (must be provided in trained_alg=False and test_cnn=True). This is sort of a
    debugging tool to make sure that samples are correctly labeled. :param max_ckpt: maximum number of checkpoints to
    keep
    """

    def __init__(self,
                 train_dataset_path=None,
                 mini_batch_size=50,
                 epochs_nbr=10,
                 learn_rate=0.001,
                 shuffle_data=True,
                 test_dataset_path=None,
                 model_pars_path=None,
                 keep_nodes_prob=0.5,
                 max_early_stop_warnings=3,
                 max_ckpt=3):

        # # # # # # # # # # # # # # # # # #  Attributes of the training loop # # # # # # # # # # # # # # # # # #
        self.train_sess = None
        self.test_sess = None
        self.shuffle_data = shuffle_data
        # Attributes related to Managing Experiments
        self.checkpoints_dir_name = "Jeong_CNN_checkpoints"
        self.checkpoints_dir = os.path.join(os.getcwd(), self.checkpoints_dir_name)
        self.pars_saver = None  # This'll hold a Saver object.
        self.max_ckpt = max_ckpt
        # # # # # Hyper-parameters of training loop of the CNN model
        self.mini_batch_size = mini_batch_size
        self.learn_rate = learn_rate
        self.keep_nodes_prob = keep_nodes_prob
        self.epochs_nbr = epochs_nbr
        # Attributes of the early stopping algorithm
        self.max_early_stop_warnings = max_early_stop_warnings
        self.cur_test_acc = 0  # current test accuracy
        self.cur_train_acc = 0  # current training accuracy
        self.max_test_acc = 0  # the maximum accuracy of the model on the test dataset since the beginning of the
        # training
        self.pre_train_acc = 0  # previous training accuracy
        self.warnings_nbr = 0  # current number of overfitting warnings. If this attribures becomes >
        # self.max_early_stop_warnings, the user is given the option to stop or continue the training
        self.early_stop_flag = False  # True means early stopping conditions are satisfied (i.e., reached the max #
        # of warnings and the user confirmed the stopping)
        # Paths and directories
        self.train_dataset_path = train_dataset_path
        self.test_dataset_path = test_dataset_path
        self.model_pars_path = model_pars_path

    def _build_train_time_cnn(self, cnn_in):
        """
        Creates the CNN model as an operation that is eventually returned
        :param cnn_in:
        :return:
        """

        # Todo: build the CNN model and put its output in 'cnn_out'
        # compute the output of the 1st conv layer
        conv1 = cnnh.conv_layer(namescope='conv1',
                                inputs=cnn_in,
                                nbr_of_filt=32,
                                filt_h=2,
                                filt_w=2,
                                stride=1,
                                padding='SAME')

        conv2 = cnnh.conv_layer(namescope='conv2',
                                inputs=conv1,
                                nbr_of_filt=32,
                                filt_h=2,
                                filt_w=2,
                                stride=1,
                                padding='SAME')

        pool1 = cnnh.pool_layer(inputs=conv2,
                                filt_h=2,
                                filt_w=2,
                                stride=2,
                                padding='VALID')

        conv3 = cnnh.conv_layer(namescope='conv3',
                                inputs=pool1,
                                nbr_of_filt=64,
                                filt_h=2,
                                filt_w=2,
                                stride=1,
                                padding='SAME')

        conv4 = cnnh.conv_layer(namescope='conv4',
                                inputs=conv3,
                                nbr_of_filt=64,
                                filt_h=2,
                                filt_w=2,
                                stride=1,
                                padding='SAME')

        pool2 = cnnh.pool_layer(inputs=conv4,
                                filt_h=2,
                                filt_w=2,
                                stride=2,
                                padding='VALID')

        # pool2_size = number of elements of each mini_batch element:
        # pool2_size is also = the number of weights per node of the next fully connecter layer
        # expected number = 4096 because:
        # one element of the mini_batch output from pool 2 has the shape [8, 8, 64].
        # thus, the total number of elements is 8 * 8 * 64 = 4096.

        # Prepare pool2 to be forward to a fully connected layer.
        # Explanation on the shape [-1, 4096]:
        # -1 is for the dynamic samples_batch size
        # 4096 is the number of elements corresponding to one sample, that is, one mini_batch element.
        # Thus number, 4096, can be obtained automatically using the following command
        # pool2.shape[1] * pool2.shape[2] * pool2.shape[3] => returns 4096 integer.

        pool2 = tf.reshape(pool2, [-1, 4096])

        # Now, each element of the mini_batch pool has the shape [1, 4096] Thus, the weight matrix of fc must be [
        # 4096, 512]; where 512 is the number of neurons in fc1 the output shape of one mini-samples_batch of fc is
        # hence: [1, 512] <= Result of multiplying two matrices of shape [1, 4096] and [4096, 512]

        fc1 = cnnh.fc_layer(namescope='fc1',
                            inputs=pool2,
                            nodes_nbr=512,
                            weights_per_node_nbr=4096)
        fc1 = tf.nn.dropout(x=fc1, rate=self.keep_nodes_prob)  # !!do not use this at test time!!

        fc2 = cnnh.fc_layer(namescope='fc2',
                            inputs=fc1,
                            nodes_nbr=512,
                            weights_per_node_nbr=512)
        fc2 = tf.nn.dropout(x=fc2, rate=self.keep_nodes_prob)  # !!do not use this at test time!!

        cnn_out = cnnh.out_layer(namescope='out',
                                 inputs=fc2,
                                 weights_per_node_nbr=512)
        return cnn_out

    def _build_test_time_cnn(self, cnn_in):
        """
        Creates the CNN model as an operation that is eventually returned
        :param cnn_in:
        :return:
        """
        # compute the output of the 1st conv layer
        conv1 = cnnh.conv_layer(namescope='conv1',
                                inputs=cnn_in,
                                nbr_of_filt=32,
                                filt_h=2,
                                filt_w=2,
                                stride=1,
                                padding='SAME')

        conv2 = cnnh.conv_layer(namescope='conv2',
                                inputs=conv1,
                                nbr_of_filt=32,
                                filt_h=2,
                                filt_w=2,
                                stride=1,
                                padding='SAME')

        pool1 = cnnh.pool_layer(inputs=conv2,
                                filt_h=2,
                                filt_w=2,
                                stride=2,
                                padding='VALID')

        conv3 = cnnh.conv_layer(namescope='conv3',
                                inputs=pool1,
                                nbr_of_filt=64,
                                filt_h=2,
                                filt_w=2,
                                stride=1,
                                padding='SAME')

        conv4 = cnnh.conv_layer(namescope='conv4',
                                inputs=conv3,
                                nbr_of_filt=64,
                                filt_h=2,
                                filt_w=2,
                                stride=1,
                                padding='SAME')

        pool2 = cnnh.pool_layer(inputs=conv4,
                                filt_h=2,
                                filt_w=2,
                                stride=2,
                                padding='VALID')

        # pool2_size = number of elements of each mini_batch element:
        # pool2_size is also = the number of weights per node of the next fully connecter layer
        # expected number = 4096 because:
        # one element of the mini_batch output from pool 2 has the shape [8, 8, 64].
        # thus, the total number of elements is 8 * 8 * 64 = 4096.

        # Prepare pool2 to be forward to a fully connected layer.
        # Explanation on the shape [-1, 4096]:
        # -1 is for the dynamic samples_batch size
        # 4096 is the number of elements corresponding to one sample, that is, one mini_batch element.
        # Thus number, 4096, can be obtained automatically using the following command
        # pool2.shape[1] * pool2.shape[2] * pool2.shape[3] => returns 4096 integer.

        pool2 = tf.reshape(pool2, [-1, 4096])

        # Now, each element of the mini_batch pool has the shape [1, 4096] Thus, the weight matrix of fc must be [
        # 4096, 512]; where 512 is the number of neurons in fc1 the output shape of one mini-samples_batch of fc is
        # hence: [1, 512] <= Result of multiplying two matrices of shape [1, 4096] and [4096, 512]

        fc1 = cnnh.fc_layer(namescope='fc1',
                            inputs=pool2,
                            nodes_nbr=512,
                            weights_per_node_nbr=4096)

        fc2 = cnnh.fc_layer(namescope='fc2',
                            inputs=fc1,
                            nodes_nbr=512,
                            weights_per_node_nbr=512)

        cnn_out = cnnh.out_layer(namescope='out',
                                 inputs=fc2,
                                 weights_per_node_nbr=512)
        return cnn_out

    def _build_smaller_cnn(self, cnn_in):
        """
        Creates the CNN model as an operation that is eventually returned
        :param cnn_in:
        :return:
        """

        # Todo: build the CNN model and put its output in 'cnn_out'
        # compute the output of the 1st conv layer
        conv1 = cnnh.conv_layer(namescope='conv1',
                                inputs=cnn_in,
                                nbr_of_filt=32,
                                filt_h=2,
                                filt_w=2,
                                stride=1,
                                padding='SAME')

        pool1 = cnnh.pool_layer(inputs=conv1,
                                filt_h=2,
                                filt_w=2,
                                stride=4,
                                padding='VALID')

        # pool2_size = number of elements of each mini_batch element:
        # pool2_size is also = the number of weights per node of the next fully connecter layer
        # expected number = 4096 because:
        # one element of the mini_batch output from pool 2 has the shape [8, 8, 64].
        # thus, the total number of elements is 8 * 8 * 64 = 4096.

        # Prepare pool2 to be forward to a fully connected layer.
        # Explanation on the shape [-1, 4096]:
        # -1 is for the dynamic samples_batch size
        # 4096 is the number of elements corresponding to one sample, that is, one mini_batch element.
        # Thus number, 4096, can be obtained automatically using the following command
        # pool2.shape[1] * pool2.shape[2] * pool2.shape[3] => returns 4096 integer.
        kk = 2048
        pool1 = tf.reshape(pool1, [-1, 2048])

        # Now, each element of the mini_batch pool has the shape [1, 4096] Thus, the weight matrix of fc must be [
        # 4096, 512]; where 512 is the number of neurons in fc1 the output shape of one mini-samples_batch of fc is
        # hence: [1, 512] <= Result of multiplying two matrices of shape [1, 4096] and [4096, 512]

        fc1 = cnnh.fc_layer(namescope='fc1',
                            inputs=pool1,
                            nodes_nbr=512,
                            weights_per_node_nbr=kk)
        fc1 = tf.nn.dropout(x=fc1, rate=self.keep_nodes_prob)  # !!do not use this at test time!!

        fc2 = cnnh.fc_layer(namescope='fc2',
                            inputs=fc1,
                            nodes_nbr=512,
                            weights_per_node_nbr=512)
        fc2 = tf.nn.dropout(x=fc2, rate=self.keep_nodes_prob)  # !!do not use this at test time!!

        cnn_out = cnnh.out_layer(namescope='out',
                                 inputs=fc2,
                                 weights_per_node_nbr=512)
        return cnn_out

    def train_cnn(self, visualize=True):
        """
        Train the convolutional neural network built using the method _build_train_time_cnn()
        :param visualize: if True, a prompt to visulize traning and test samples with corresponding labels pops-up.
        """
        self._remove_dir_files(f_dir=self.checkpoints_dir)  # remove all checkpoint files
        self._remove_dir_files(
            f_dir=os.path.join(os.getcwd(), 'Jeong_CNN_graphs'))  # remove all event files (used for summary)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # get the training data as batches class
        self.train_data_as_mini_batches = self.get_data(data_path=self.train_dataset_path,
                                                        mini_batch_size=self.mini_batch_size,
                                                        visualize=visualize,
                                                        shuffle=True,
                                                        vis_mssg="Visualizing the training dataset")
        # get the tuple for ops to get the next training_samples_batch and to initialize ITS iterator object.
        self.get_next_train_mini_batch_op, self.train_set_iter_init_op = self.get_next_element(
            iterable_data=self.train_data_as_mini_batches)
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # get the test data as batches class
        self.test_data_as_mini_batches = self.get_data(data_path=self.test_dataset_path,
                                                       mini_batch_size=self.mini_batch_size,
                                                       visualize=visualize,
                                                       horz_flip=False,  # Do not augment test data
                                                       shuffle=False,  # No need to shuffle the data
                                                       vis_mssg="Visualizing the test dataset")
        # get the tuple for ops to get the next test_samples_batch and to initialize ITS iterator object.
        self.get_next_test_mini_batch_op, self.test_set_iter_init_op = self.get_next_element(
            iterable_data=self.test_data_as_mini_batches)
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # create a placeholder as the hl_alg_input of the cnn classifier
        self.cnn_in_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3],
                                                  name="CNN_input_placeholder")
        # create a placeholder for labels. This placeholder is involved in the loss computation
        self.labels_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 1])

        # Creating the op computing the cnn's output. Variables are declared within this op.
        # WARNING: Make sure this op is created before the op initializing variables.
        trn_t_cnn_op = self._build_train_time_cnn(self.cnn_in_ph)  # TRaiN_Time_CNN_OPeration\
        # These two graphs can share the same hl_alg_input placeholder because they are run one after another.
        self.tst_t_cnn_op = self._build_test_time_cnn(self.cnn_in_ph)  # TeST_Time_CNN_OPeration /
        one_training_step_op, cost_op = self._one_train_step(predictions=trn_t_cnn_op, labels=self.labels_ph)

        # WARNING: Make sure the variable initialization op is created before declaring variables
        var_init_op = tf.compat.v1.global_variables_initializer()

        # # # # # # # # # # # # # # # # TENSORBOARD SUMMARY # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # create placeholders to hl_alg_input training and test accuracy to summary ops
        cur_train_acc_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=())
        cur_test_acc_ph = tf.compat.v1.placeholder(dtype=tf.float32,
                                                   shape=())  # shape=() corresponds to rank 0 tensor, i.e., a scalar

        tf.compat.v1.summary.scalar('Training Accuracy', tensor=cur_train_acc_ph)
        tf.compat.v1.summary.scalar('Test Accuracy', tensor=cur_test_acc_ph)
        summary_op = tf.compat.v1.summary.merge_all()  # merge ops that would have been returned by previous two
        # statements
        sumr_writer = tf.compat.v1.summary.FileWriter(logdir='./Jeong_CNN_graphs',
                                                      graph=tf.compat.v1.get_default_graph())  # create a summary
        # sumr_writer for the default graph
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        self._create_pars_saver(max_to_keep=self.max_ckpt)
        with tf.compat.v1.Session() as self.train_sess:
            self.train_sess.run(var_init_op)
            self.train_sess.run(self.train_set_iter_init_op)

            # INITIALIZE THE PARAMETERS OF THE EARLY STOPPING ALGORITHM
            self.max_test_acc = 0
            self.pre_train_acc = 0
            self.warnings_nbr = 0

            # The main training loop: performs self.epochs_nbr full cycles on the entire training dataset.
            # for cur_epoch in range(0, self.epochs_nbr):
            self.cur_epoch = 0
            self.cur_train_step = 0  # current training step (or iteration)
            while True:
                if self.early_stop_flag:
                    break
                try:
                    # This while loop iterates over all mini-batches of the training dataset
                    self.cur_minibatch_order = 0
                    while True:  # True as long as there is an element from the dataset
                        if self.early_stop_flag:
                            break
                        self.train_samples_mini_batch, self.train_labels_mini_batch = self.train_sess.run(
                            fetches=self.get_next_train_mini_batch_op)
                        _, self.train_predictions, cost = self.train_sess.run(
                            fetches=[one_training_step_op, trn_t_cnn_op, cost_op],
                            feed_dict={self.cnn_in_ph: self.train_samples_mini_batch,
                                       self.labels_ph: self.train_labels_mini_batch})
                        # self.cur_train_acc = cnnh.model_accuracy(predictions=self.train_predictions,
                        #                                          labels=self.train_labels_mini_batch)
                        # print("At {}^th training step, {}^th mini-batch, training accuracy on current mini-batch ={}%"
                        #       .format(self.cur_train_step, self.cur_minibatch_order, self.cur_train_acc))
                        self.cur_minibatch_order += 1
                        self.cur_train_step += 1

                except tf.errors.OutOfRangeError:  # This error raises when there is no next element to get; in other
                    # words, one cur_epoch has been completed IMPORTANT NOTE FOR DEVELOPER: IF YOU GET ALWAYS A
                    # CONSTANT ACCURACY (LIKE 50%), IT LIKELY MEANS THAT OUTPUT PREDICTIONS ARE ALL nan.

                    # After raising the error 'tf.errors.OutOfRangeError', the next cur_epoch won't be run unless the
                    # iterator is initialized so the first element (i.e., mini-samples_batch) can be extracted again.
                    self.compute_train_set_predictions()
                    self.cur_train_acc = cnnh.model_accuracy(predictions=self.train_predictions,
                                                             labels=self.train_labels)

                    self.compute_test_set_predictions()
                    self.cur_test_acc = cnnh.model_accuracy(predictions=self.test_predictions,
                                                            labels=self.test_labels)

                    print(
                        "\nThe {}^th EPOCH has been finished with the training accuracy = {}% AND the test accuracy = "
                        "{}% \n".format(
                            round(self.cur_epoch, 4),
                            round(self.cur_train_acc, 4),
                            round(self.cur_test_acc, 4)))

                    # EARLY STOPPING ALGORITHM: sets the attribute self.early_stop_flag to True in case all early
                    # stopping conditions are satisfied
                    self.regularize_early_stop()
                    # # # # # # # # write one summary point per epoch # # # # # # # # # # # #
                    # get protobuf data and put it into 'summary'
                    summary = self.train_sess.run(fetches=summary_op,
                                                  feed_dict={cur_train_acc_ph: self.cur_train_acc,
                                                             cur_test_acc_ph: self.cur_test_acc})
                    # add protobuf data to event file (used by TensorBoard server)
                    sumr_writer.add_summary(summary=summary, global_step=self.cur_epoch)
                    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
                    self.cur_epoch += 1

                    # After raising the error 'tf.errors.OutOfRangeError', the next cur_epoch won't be run unless the
                    # iterator is initialized so the first element (i.e., mini-samples_batch) can be extracted again.
                    self.train_sess.run(self.train_set_iter_init_op)

                # Stop the training if the output of the CNN is nan (not a number)
                ################################################################################################
                graph_vars = tf.compat.v1.get_collection(key=tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
                out_layer_tensor = graph_vars[-1]
                cnn_output = self.train_sess.run(out_layer_tensor)
                if isnan(cnn_output[0]):  # True if the output is nan
                    messagebox.showwarning("Warning",
                                           message="The CNN's output is nan and the training loop will be exit\n"
                                                   "Potenial cause of 'nan' are exploding gradients")
                    print("The training has been stopped due to nan value in the output")
                    break
                ################################################################################################

            # close the summary sumr_writer
            sumr_writer.close()

    def _one_train_step(self, predictions, labels):
        """
        create and return an optimization op that runs one step of training
        :param predictions:  predictions samples_batch
        :param labels: labels samples_batch
        :return:
        """
        # The expected shape is (number_of_samples, loss_value_of_the_i^th_sample)
        loss_op = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=predictions)
        # The cost function is the mean value of losses of all samples
        cost_op = tf.reduce_mean(loss_op)
        #  one_train_step_op = tf.compat.v1.train.AdamOptimizer().minimize(cost)
        one_train_step_op = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.learn_rate).minimize(cost_op)
        return one_train_step_op, cost_op

    def regularize_early_stop(self):
        """
        This algorithm stops the training if the number of warnings exceeds the threshold self.warnings_nbr.
        The number of warning is incremented if two conditions are satisfied:
            1) the current test accuracy is higher than the maximum test accuracy achieved during previous epochs
            2) the training accuracy of the current epoch is higher than the training accuracy of the previous epoch
               OR
               the training accuracy of the current epoch exceeds 99%
        :return: True if if all early stopping conditions are satisfied and the user confirms the early stoping from the
                 console. If the user does not confirm the early stopping, the number of early stopping warnings is reset
                 to zero.
        """
        self.early_stop_flag = False
        if self.cur_test_acc < self.max_test_acc:  # first warning condition
            if (self.cur_train_acc > self.pre_train_acc) or (self.cur_train_acc > 98):  # second warning condition
                # increment the number of overfitting warnings if the test accuracy did not improve and the training
                # accuracy did.
                self.warnings_nbr += 1
                if self.warnings_nbr == self.max_early_stop_warnings:  # True means the model is overfitting: ask the
                    # user whether he wants to continue training:
                    print("The number of overfitting warnings reached {} warnings".format(self.warnings_nbr))
                    text = "The maximum test accuracy achieved is: {}%. The corresponding training accuracy is: " \
                           "{}%\n".format(round(self.max_test_acc, 4), round(self.train_acc_of_max_text_acc, 4))
                    print(text)

                    # After detecting an overfitting, request user permission to stop the training.
                    stop_training = ""
                    while stop_training != 'y' and stop_training != 'n':
                        stop_training = input("DO YOU WANT TO STOP THE TRAINING LOOP (y/n)?")
                    if stop_training == 'y':
                        self.early_stop_flag = True
                    else:
                        self.warnings_nbr = 0

            else:  # no overfitting warning: only the first warning condition is satisfied
                self.warnings_nbr = 0  # reset the number of warnings to zero
        else:  # no overfitting warning
            # implies the test accuracy is better than the last maximum test accuracy. Thus, three actions are taken:
            # action 1: update the maximum test accuracy self.max_test_acc
            self.max_test_acc = self.cur_test_acc
            self.train_acc_of_max_text_acc = self.cur_train_acc
            # action 2: reset the number of warnings warnings_nbr to zero
            self.warnings_nbr = 0  # reset the number of warnings to zero

            # action 3: save the model (we want to save the model with the highest test accuracy).
            self._save_cnn_pars()

        # the cur_train_acc of this cur_epoch is the self.pre_train_acc of the next cur_epoch
        self.pre_train_acc = self.cur_train_acc
        ################################################################################################
        # END OF EARLY STOPPING ALGORITHM
        return self.early_stop_flag

    def get_data(self, data_path, mini_batch_size, horz_flip=True, normalize=True, visualize=True, shuffle=True,
                 vis_mssg="No Message"):
        # Shuffling using TF API is much more efficient. Hence the argument value shuffle_data=False
        data_as_mini_batches = cnnh.create_batch_as_np(data_path=data_path, shuffle_data=False)
        if horz_flip:
            data_as_mini_batches = cnnh.augment_batch(batch=data_as_mini_batches)
        if normalize:
            data_as_mini_batches = cnnh.normalize_batch(batch=data_as_mini_batches)
        if visualize:
            print(vis_mssg)
            cnnh.visualize_dataset(batch=data_as_mini_batches)

        number_of_samples = data_as_mini_batches[0].shape[0]
        data_as_mini_batches = tf.data.Dataset.from_tensor_slices(data_as_mini_batches)
        if shuffle:
            data_as_mini_batches = data_as_mini_batches.shuffle(buffer_size=number_of_samples)
        # Transform dataset as mini-batches
        data_as_mini_batches = data_as_mini_batches.batch(batch_size=mini_batch_size)

        return data_as_mini_batches

    def get_next_element(self, iterable_data):
        """
        Creates and returns two operations:
            the first is an op configured to get the next element of the dataset given in 'iterable_data'.
            the second is to initialize the iterator object associated to the first op.

        NOTE: if elements of 'iterable_data' are mini_batches, the op 'get_next_element_op' would obviously return
        the next mini_batch, not the next datapoint and its label.

        :param iterable_data: the data you want to iterate over
        :return: a tuple (get_next_element_op, iter_init_op) representing the first op and second op, respectively.
        """
        # Create an iterator op and configure it on the dataset 'iterable_data'
        iter_op = tf.compat.v1.data.make_initializable_iterator(iterable_data)
        # Run this op before iter_op
        iter_init_op = iter_op.initializer
        # Create an op that gets the next element (e.g., samples_batch) of the dataset given in 'iterable_data'
        get_next_element_op = iter_op.get_next()
        return get_next_element_op, iter_init_op

    def _create_pars_saver(self, max_to_keep=2, keep_checkpoint_every_n_hours=1):
        """
        Create an instance of the class Saver (Full class name tf.compat.v1.train.Saver)
        :param max_to_keep: the maximum number of checkpoints to keep in 'self.checkpoints_dir'
        :param keep_checkpoint_every_n_hours: MAX number of new checkpoints to keep per ONE hour of training
        :return:
        """
        if self.pars_saver is None:
            all_scopes_var_list = []
            for scope_name in cts.PARS_SCOPE_NAMES:  # 'scope_name' would be a list of variables of a given name scope
                # of the list 'pars_scope_names'.
                one_scope_var_list = tf.compat.v1.get_collection(
                    key=tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                    scope=scope_name)
                for tf_var_ob in one_scope_var_list:  # 'tf_var_ob' is one single TF variable object associated
                    # with a given parameter of the CNN model.
                    all_scopes_var_list.append(tf_var_ob)

            self.pars_saver = tf.compat.v1.train.Saver(var_list=all_scopes_var_list,
                                                       max_to_keep=max_to_keep,
                                                       keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)

    def _save_cnn_pars(self, checkpoint_files_name=None):
        """
        Save the model's parameters corresponding to the highest test accuracy.
        :param checkpoint_files_name: name of checkpoint files
        :return:
        """
        if checkpoint_files_name is None:
            # Prefix of filenames created for the checkpoint of checkpoint files: TR_acc <=> TRaining accuracy;
            # TS_acc <=> TeSt accuracy; step <=> order of the training step corresponding to TR_acc and TS_acc
            checkpoint_files_name = "CKPT_TR_acc_{}_TS_acc_{}_Epoch_{}".format(round(self.cur_train_acc, 3),
                                                                               round(self.cur_test_acc, 3),
                                                                               self.cur_epoch)

        save_path = os.path.join(self.checkpoints_dir, checkpoint_files_name)
        # create an empty list to be filled with TF variables associated with the CNN's parameters

        self.pars_saver.save(sess=self.train_sess, save_path=save_path)  # global_step=int(self.cur_train_step)

    def _remove_dir_files(self, f_dir):
        """
        Removes all files inside a directory
        :param f_dir: directory holding files to remove
        """
        if os.path.exists(f_dir):
            for filename in os.listdir(f_dir):
                filepath = os.path.join(f_dir, filename)
                os.remove(filepath)

    def _restore_cnn_pars(self, path, session):
        """
        Restore the parameters (TF variables) of the built cnn model :param path: path to the checkpoint file (
        contains saved parameters as TF.Variables) (e.g., checkpoints/dummy_model-49) :param session: the session
        object to restore the saved parameters into
        """
        # # # # # # load variables # # # # # # # # # # # # #
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess=session, save_path=path)
        # # # # # # # # # # # # # # # # # # # # # # # # # #

    def get_early_stop_data(self):
        """
        DEPRECATED.
        :return:
        """
        pass
        # """
        # Gets early stop train and test dataset as batches. Recall that after an epoch, evaluation of accuracy on the
        # entire training and test dataset is required.
        # :return:
        # """
        # self.train_samples, self.train_labels = cnnh.create_batch_as_np(self.train_dataset_path)
        # self.train_samples, self.train_labels = cnnh.augment_batch((self.train_samples, self.train_labels))
        # self.train_samples, self.train_labels = cnnh.normalize_batch((self.train_samples, self.train_labels))
        #
        # self.test_samples, self.test_labels = cnnh.create_batch_as_np(self.test_dataset_path)
        # self.test_samples, self.test_labels = cnnh.normalize_batch(batch=(self.test_samples, self.test_labels))

    def compute_train_set_predictions(self):
        """
        Description:
        -----------
        Computes predictions on the entire training set, which's been previously sliced into mini-batches by the method
        batch(). This method is useful for the early stopping algorithm.

        NOTE:
        -----
        Slicing into mini-batches is useful in that the data loaded into the input placeholder the classifier's
        graph does not exceed available RAM.
        :return:
        """
        # Initialize the mini-batch iterator
        self.train_sess.run(fetches=self.train_set_iter_init_op)
        # Initialize arrays of training predictions and training labels
        # Note: training_labels is a redundant process that can be avoided! Not a big deal+I'm in a hurry now + I'm Lazy
        self.train_predictions = np.array([])
        self.train_labels = np.array([])
        try:  # try to get the next mini-batch
            while True:
                # Get training mini-batch samples and labels
                self.train_samples_mini_batch, self.train_labels_mini_batch = self.train_sess.run(
                    fetches=self.get_next_train_mini_batch_op)
                # append labels of the current mini-batch
                self.train_labels = np.append(arr=self.train_labels, values=self.train_labels_mini_batch)
                # compute predictions
                predictions_to_append = self.train_sess.run(fetches=self.tst_t_cnn_op,
                                                            feed_dict={self.cnn_in_ph: self.train_samples_mini_batch})
                self.train_predictions = np.append(arr=self.train_predictions, values=predictions_to_append)
        except tf.errors.OutOfRangeError:  # executes if all mini-batches have been iterated over
            rows = self.train_predictions.size
            cols = 1
            if rows != self.train_labels.size:
                err_msg = "The size of self.train_predictions (={}) and the size of self.train_labels (={}) do not " \
                          "match" \
                    .format(self.train_predictions, self.train_labels)
                raise RuntimeError(err_msg)
            self.train_predictions = np.reshape(a=self.train_predictions, newshape=(rows, cols))
            self.train_labels = np.reshape(a=self.train_labels, newshape=(rows, cols))

    def compute_test_set_predictions(self):
        """
        Description:
        -----------
        Computes predictions on the entire test set, which's been previously sliced into mini-batches by the method
        batch(). This method is useful for the early stopping algorithm.

        NOTE:
        -----
        Slicing into mini-batches is useful in that the data loaded into the input placeholder the classifier's
        graph does not exceed available RAM.
        """

        # Initialize the mini-batch iterator
        self.train_sess.run(fetches=self.test_set_iter_init_op)
        # Initialize the predictions array (final shape would be: (number_of_test_samples, 1)
        self.test_predictions = np.array([])
        self.test_labels = np.array([])
        try:  # try to get the next mini-batch
            while True:
                # Get test mini-batch samples and labels
                self.test_samples_mini_batch, self.test_labels_mini_batch = self.train_sess.run(
                    fetches=self.get_next_test_mini_batch_op)
                # append labels of the current mini-batch
                self.test_labels = np.append(arr=self.test_labels, values=self.test_labels_mini_batch)
                # compute predictions on the current test samples mini-batch
                predictions_to_append = self.train_sess.run(fetches=self.tst_t_cnn_op,
                                                            feed_dict={self.cnn_in_ph: self.test_samples_mini_batch})
                self.test_predictions = np.append(arr=self.test_predictions, values=predictions_to_append)
        except tf.errors.OutOfRangeError:  # executes if all mini-batches have been iterated over
            rows = self.test_predictions.size
            cols = 1
            if rows != self.test_labels.size:
                err_msg = "The size of self.train_predictions (={}) and the size of self.train_labels (={}) do not " \
                          "match" \
                    .format(self.test_predictions, self.test_labels)
                raise RuntimeError(err_msg)
            self.test_predictions = np.reshape(a=self.test_predictions, newshape=(rows, cols))
            self.test_labels = np.reshape(a=self.test_labels, newshape=(rows, cols))


class JeongTrainedCnn(CnnTrainer):
    def __init__(self, trained_pars_path,
                 session: tf.compat.v1.Session()):  # I do not need to run the initialization method of the parent class
        """
        :param trained_pars_path: path to the model's parameters to load :param session: the session to load the
        model's parameters into and to classify patches (see the method _classify_patches)
        """
        self.sess = session
        # Get the test time classifier (no dropout) as a graph
        self.cnn_in_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3],
                                                  name="CNN_input_placeholder")
        self.cnn_graph_op = self._build_test_time_cnn(cnn_in=self.cnn_in_ph)

        # Restore the previously trained model
        self._restore_cnn_pars(path=trained_pars_path, session=self.sess)

    def classify_patches(self, patches_batch):
        """
        A method that classifies patches.
        :param patches_batch: patches to classify, stacked into a numpy array of shape (number_of_patches, 32, 32, 3)
        """
        self.cnn_out = self.sess.run(fetches=self.cnn_graph_op, feed_dict={self.cnn_in_ph: patches_batch})
        # positive and negative values of the current value of the attribute self.cnn_out correspond to class 1 and
        # class 0, respectively, because  a sigmoid function of positive values is > 0.5 (i.e., 50% probability of
        # being class 1). returns a copy of self.cnn_out, where all elements satisfying the condition are replaced by
        # 1. Other elements are replaced by 0.
        self.cnn_out = np.where(self.cnn_out > 0, 1, 0)
        return self.cnn_out


class JeongPixs:
    def __init__(self, min_hyst_th=50, max_hyst_th=130, med_scales_nbr=5, t=153):
        self.min_hyst_th = min_hyst_th
        self.max_hyst_th = max_hyst_th
        self.med_scales_nbr = med_scales_nbr  # number of applied median filters
        # self.pixs_xy = []  # [(x1,y1), (x2,y2), ...]
        self.t = t  # threshold applied on the weighted edge map (see equation 3 in the paper)
        self.src_im_g = None
        self.median_im = None
        self.canny_im = None
        self.W = None
        self.W_T = None
        self.cand_pixs_xy = None
        self.in_frame_cand_pixs_xy = None

    def get_cand_pixs(self, src_im):
        """
        Finds candidat edge pixels in src_im. :param src_im: the source image to find candidate edge pixels in
        :return: a tuple self.in_frame_cand_pixs_xy, self.W_T: * self.W_T is the final binary image containing
        candidate edge pixels * self.in_frame_cand_pixs_xy is the tuple (cand_pixs_x, y); where: cand_pixs_x is an
        int64 numpy array containing all cand_pixs_x coordinates of in-frame candidate pixels and, y is an int64
        numpy array containing all y coordinates of in-frame candidate pixels
        """
        self.src_im_g = cv.cvtColor(src_im, cv.COLOR_BGR2GRAY)
        w, h = self.src_im_g.shape
        self.median_im = self.canny_im = np.array(np.zeros((self.med_scales_nbr, w, h)))
        for s in range(0, self.med_scales_nbr):
            ksize = (10 * (s + 1)) + 1  # Kernel size
            self.median_im[s] = cv.medianBlur(self.src_im_g, ksize)
            self.canny_im[s] = cv.Canny(np.uint8(self.median_im[s]),
                                        threshold1=self.min_hyst_th,
                                        threshold2=self.max_hyst_th)
        # Computed the weighted edge map W_T. W_T represent the final detected edges
        self.W = np.mean(self.canny_im,
                         axis=0)  # axis = 0 corresponds to axis along scales_nbr, which corresponds to scales
        ret, self.W_T = cv.threshold(self.W, self.t, 255,
                                     0)  # cv.threshold(W, threshold, replace_pixels_with_this_value, cand_x
        # threshold parameter)
        self.cand_pixs_xy = np.where(
            self.W_T == 255)  # self.cand_pixs_xy = (np.arr of y coordinates, np.arr of cand_pixs_x coordinates)
        self.cand_pixs_xy = self.cand_pixs_xy[1], self.cand_pixs_xy[
            0]  # Now, self.cand_pixs_xy = (np.arr of cand_pixs_x coordinates, np.arr of y coordinates)
        x, y = self.cand_pixs_xy
        self.in_frame_cand_pixs_xy = self.in_frame_xy(x=x,
                                                      y=y,
                                                      patch_w=32,
                                                      patch_h=32,
                                                      frame_w=w,
                                                      frame_h=h)
        return self.cand_pixs_xy, self.W_T

    def in_frame_xy(self, x, y, patch_w, patch_h, frame_w, frame_h):
        """
        1) Description: Keep only center coordinates that correspond to in-frame patches. An in-frame patch is patch
        with an intersection with the image from which it is taken that's equal to its surface (i.e.,
        number of pixels). 2) How it works: Delete all patches that are out-of-frame (i.e., not in-frame). Such
        patches have center coordinates xy with cand_pixs_x ∉ [xmin, xmax] and/or y ∉ [ymin, ymax]. (see computation
        of xmin, xmax, ymin, ymax below) Thus, to delete out-of-frame patches and keep only in-frame patches,
        this method deletes all center coordinates (cand_pixs_x and corresponding y) of patches whose cand_pixs_x ∉ [
        xmin, xmax] or y ∉ [ymin, ymax].

            Computation of xmin, xmax, ymin, ymax: Consider patch_w, patch_h, frame_w and frame_h (see description in
            parameters description) To understand how xmin, xmax, ymin, ymax are computed, you must understand that
            extraction of each patch from an image requires two points: starting point ps = (xs, ys) and ending point
            pe = (xe, ye). A patch is extracted from an image I by slicing it using ps and pe: I[ys:ye,
            xs:xe]. Computation of ps and pe: equation 1: xs = cand_pixs_x[i] - int(patch_w / 2); where cand_pixs_x[
            i] is the cand_pixs_x coordinate of the center of the i^th patch equation 2: xe = xs + patch_w equation
            3: ys = y[i] - int(patch_h / 2); where y[i] is the y coordinate of the center of the i^th patch equation
            4: ye = ys + patch_h

            It is obvious that for the extracted patch to be an in-frame patch, the follwing conditions must be
            satisfied: condition 1: xs ≥ 0, ys ≥ 0 condition 2: xe ≤ frame_w - 1, ye ≤ frame_h - 1 The problem to
            solve is to conclude what values of cand_pixs_x[i] and y[i] (xy coordinates of the i^th patch) would
            allow satisfaction of condition 1 and 2. This can easily be inferred from equations given by: equation1,
            2, 3, 4, and condition 1 and 2: xs ≥ 0 and xs = cand_pixs_x[i] - int(patch_w / 2) ==> cand_pixs_x[i] ≥
            int(patch_w / 2) ys ≥ 0 and ys = cand_pixs_x[i] - int(patch_h / 2) ==> y[i] ≥ int(patch_h / 2)

                xe ≤ frame_w - 1, xe = xs + patch_w and xs = cand_pixs_x[i] - int(patch_w /2) ==> cand_pixs_x[i] ≤
                frame_w - 1 + int(patch_w / 2) - patch_w ye ≤ frame_h - 1, ye = ys + patch_h and ys = y[i] - int(
                patch_h /2) ==> y[i] ≤ frame_h - 1 + int(patch_h / 2) - patch_h

                Thus: int(patch_w/ 2) ≤ cand_pixs_x[i] ≤ frame_w - 1 + int(patch_w / 2) - patch_w <==> xmin ≤
                cand_pixs_x[i] ≤ xmax and int(patch_h/ 2) ≤ y[i] ≤ frame_h - 1 + int(patch_h / 2) - patch_h <==> ymin
                ≤ y[i] ≤ ymax

                Finnaly:
                    xmin = int(patch_w/ 2)
                    xmax = frame_w - 1 + int(patch_w / 2) - patch_w
                    ymin = int(patch_h/ 2)
                    ymax = frame_h - 1 + int(patch_h / 2) - patch_h

        :param x: cand_pixs_x coordinates of patches centers. Must be a numpy array with shape = (1, N); N: number of
        patches :param y: y coordinates of patches centers. Must be a numpy array with shape = (1, N); N: number of
        patches :param patch_w: width of patches :param patch_h: height of patches :param frame_w: width of the frame
        (or image) from which patches are extracted :param frame_h: height of the frame (or image) from which patches
        are extracted :return: a tuple (in_fr_x, in_fr,y) that correspond to center coordinates of in-frame patches.
        """
        # compute xmin, xmax, ymin, ymax
        # debugging:
        xmin = int(patch_w / 2)
        xmax = frame_w - 1 + int(patch_w / 2) - patch_w

        ymin = int(patch_h / 2)
        ymax = frame_h - 1 + int(patch_h / 2) - patch_h
        # find indexes of cand_pixs_x not in [xmin, xmax], i.e., indexes of cand_pixs_x that where an element is
        # smaller than xmin OR bigger than xmax.
        x_out_ind = np.union1d(np.where(xmin > x), np.where(x > xmax))
        # find indexes of y not in [ymin, ymax], i.e., indexes of y that where an element is smaller than ymin OR
        # bigger than ymax.
        y_out_ind = np.union1d(np.where(ymin > y), np.where(y > ymax))  # the union is equivalent to OR.

        # indexes to delete xy coordinates of out of frame patches =  similar indexes in x_out_idx and y_out_idx
        to_del_idx = np.union1d(x_out_ind, y_out_ind)  # the union removes redundant indexes
        in_x = np.delete(arr=x, obj=to_del_idx, axis=0)
        in_y = np.delete(arr=y, obj=to_del_idx, axis=0)

        return in_x, in_y
