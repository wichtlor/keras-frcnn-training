from __future__ import absolute_import
import random
import pickle
import pprint
import time
import numpy as np
from optparse import OptionParser
import traceback
import os

from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.optimizers import SGD, Adam
from keras.utils import Sequence
from keras_frcnn import config
from keras_frcnn.pascal_voc_parser import get_data
from keras_frcnn import losses as losses

from visualization.plots import save_plots_from_history
from module import data_generators



import cv2

import copy

import threading
import itertools

from keras_frcnn import roi_helpers as roi_helpers
from keras_frcnn import data_augment




def union(au, bu, area_intersection):
    area_a = (au[2] - au[0]) * (au[3] - au[1])
    area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
    area_union = area_a + area_b - area_intersection
    return area_union


def intersection(ai, bi):
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    if w < 0 or h < 0:
        return 0
    return w*h


def iou(a, b):
    # a and b should be (x1,y1,x2,y2)

    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    area_i = intersection(a, b)
    area_u = union(a, b, area_i)

    return float(area_i) / float(area_u + 1e-6)


def get_new_img_size(width, height, img_min_side=600):
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = img_min_side
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = img_min_side

    return resized_width, resized_height


class SampleSelector:
    def __init__(self, class_count):
        # ignore classes that have zero samples
        self.classes = [b for b in class_count.keys() if class_count[b] > 0]
        self.class_cycle = itertools.cycle(self.classes)
        self.curr_class = next(self.class_cycle)

    def skip_sample_for_balanced_class(self, img_data):

        class_in_img = False

        for bbox in img_data['bboxes']:

            cls_name = bbox['class']

            if cls_name == self.curr_class:
                class_in_img = True
                self.curr_class = next(self.class_cycle)
                break

        if class_in_img:
            return False
        else:
            return True


def calc_rpn(C, img_data, width, height, resized_width, resized_height, img_length_calc_function):

    downscale = float(C.rpn_stride)
    anchor_sizes = C.anchor_box_scales
    anchor_ratios = C.anchor_box_ratios
    num_anchors = len(anchor_sizes) * len(anchor_ratios)    

    # calculate the output map size based on the network architecture

    (output_width, output_height) = img_length_calc_function(resized_width, resized_height)

    n_anchratios = len(anchor_ratios)
    
    # initialise empty output objectives
    y_rpn_overlap = np.zeros((output_height, output_width, num_anchors))
    y_is_box_valid = np.zeros((output_height, output_width, num_anchors))
    y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4))

    num_bboxes = len(img_data['bboxes'])

    num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)
    best_anchor_for_bbox = -1*np.ones((num_bboxes, 4)).astype(int)
    best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)
    best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int)
    best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)

    # get the GT box coordinates, and resize to account for image resizing
    gta = np.zeros((num_bboxes, 4))
    for bbox_num, bbox in enumerate(img_data['bboxes']):
        # get the GT box coordinates, and resize to account for image resizing
        gta[bbox_num, 0] = bbox['x1'] * (resized_width / float(width))
        gta[bbox_num, 1] = bbox['x2'] * (resized_width / float(width))
        gta[bbox_num, 2] = bbox['y1'] * (resized_height / float(height))
        gta[bbox_num, 3] = bbox['y2'] * (resized_height / float(height))
    
    # rpn ground truth

    for anchor_size_idx in range(len(anchor_sizes)):
        for anchor_ratio_idx in range(n_anchratios):
            anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]
            anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]    
            
            for ix in range(output_width):                    
                # x-coordinates of the current anchor box    
                x1_anc = downscale * (ix + 0.5) - anchor_x / 2
                x2_anc = downscale * (ix + 0.5) + anchor_x / 2    
                
                # ignore boxes that go across image boundaries                    
                if x1_anc < 0 or x2_anc > resized_width:
                    continue
                    
                for jy in range(output_height):

                    # y-coordinates of the current anchor box
                    y1_anc = downscale * (jy + 0.5) - anchor_y / 2
                    y2_anc = downscale * (jy + 0.5) + anchor_y / 2

                    # ignore boxes that go across image boundaries
                    if y1_anc < 0 or y2_anc > resized_height:
                        continue

                    # bbox_type indicates whether an anchor should be a target 
                    bbox_type = 'neg'

                    # this is the best IOU for the (x,y) coord and the current anchor
                    # note that this is different from the best IOU for a GT bbox
                    best_iou_for_loc = 0.0

                    for bbox_num in range(num_bboxes):
                        
                        # get IOU of the current GT box and the current anchor box
                        curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], [x1_anc, y1_anc, x2_anc, y2_anc])
                        # calculate the regression targets if they will be needed
                        if curr_iou > best_iou_for_bbox[bbox_num] or curr_iou > C.rpn_max_overlap:
                            cx = (gta[bbox_num, 0] + gta[bbox_num, 1]) / 2.0
                            cy = (gta[bbox_num, 2] + gta[bbox_num, 3]) / 2.0
                            cxa = (x1_anc + x2_anc)/2.0
                            cya = (y1_anc + y2_anc)/2.0

                            tx = (cx - cxa) / (x2_anc - x1_anc)
                            ty = (cy - cya) / (y2_anc - y1_anc)
                            tw = np.log((gta[bbox_num, 1] - gta[bbox_num, 0]) / (x2_anc - x1_anc))
                            th = np.log((gta[bbox_num, 3] - gta[bbox_num, 2]) / (y2_anc - y1_anc))
                        
                        if img_data['bboxes'][bbox_num]['class'] != 'bg':

                            # all GT boxes should be mapped to an anchor box, so we keep track of which anchor box was best
                            if curr_iou > best_iou_for_bbox[bbox_num]:
                                best_anchor_for_bbox[bbox_num] = [jy, ix, anchor_ratio_idx, anchor_size_idx]
                                best_iou_for_bbox[bbox_num] = curr_iou
                                best_x_for_bbox[bbox_num,:] = [x1_anc, x2_anc, y1_anc, y2_anc]
                                best_dx_for_bbox[bbox_num,:] = [tx, ty, tw, th]

                            # we set the anchor to positive if the IOU is >0.7 (it does not matter if there was another better box, it just indicates overlap)
                            if curr_iou > C.rpn_max_overlap:
                                bbox_type = 'pos'
                                num_anchors_for_bbox[bbox_num] += 1
                                # we update the regression layer target if this IOU is the best for the current (x,y) and anchor position
                                if curr_iou > best_iou_for_loc:
                                    best_iou_for_loc = curr_iou
                                    best_regr = (tx, ty, tw, th)

                            # if the IOU is >0.3 and <0.7, it is ambiguous and no included in the objective
                            if C.rpn_min_overlap < curr_iou < C.rpn_max_overlap:
                                # gray zone between neg and pos
                                if bbox_type != 'pos':
                                    bbox_type = 'neutral'

                    # turn on or off outputs depending on IOUs
                    if bbox_type == 'neg':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                    elif bbox_type == 'neutral':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                    elif bbox_type == 'pos':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        start = 4 * (anchor_ratio_idx + n_anchratios * anchor_size_idx)
                        y_rpn_regr[jy, ix, start:start+4] = best_regr

    # we ensure that every bbox has at least one positive RPN region

    for idx in range(num_anchors_for_bbox.shape[0]):
        if num_anchors_for_bbox[idx] == 0:
            # no box with an IOU greater than zero ...
            if best_anchor_for_bbox[idx, 0] == -1:
                continue
            y_is_box_valid[
                best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios *
                best_anchor_for_bbox[idx,3]] = 1
            y_rpn_overlap[
                best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios *
                best_anchor_for_bbox[idx,3]] = 1
            start = 4 * (best_anchor_for_bbox[idx,2] + n_anchratios * best_anchor_for_bbox[idx,3])
            y_rpn_regr[
                best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], start:start+4] = best_dx_for_bbox[idx, :]

    y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))
    y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)

    y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
    y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)

    y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
    y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)

    pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
    neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))

    num_pos = len(pos_locs[0])

    # one issue is that the RPN has many more negative than positive regions, so we turn off some of the negative
    # regions. We also limit it to 256 regions.
    num_regions = 256

    if len(pos_locs[0]) > num_regions/2:
        val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - num_regions/2)
        y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
        num_pos = num_regions/2

    if len(neg_locs[0]) + num_pos > num_regions:
        val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos)
        y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0

    y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)
    y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1)

    return np.copy(y_rpn_cls), np.copy(y_rpn_regr)


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return next(self.it)        

    
def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

def select_rois_for_detection(Y1, C):
    #Background oder Objekt
    neg_samples = np.where(Y1[0, :, -1] == 1)
    pos_samples = np.where(Y1[0, :, -1] == 0)
    if len(neg_samples) > 0:
        neg_samples = neg_samples[0]
    else:
        neg_samples = []
    if len(pos_samples) > 0:
        pos_samples = pos_samples[0]
    else:
        pos_samples = []
    #Background und Objekt RoIs werden ausgewaehlt und ergeben die Batch fuer den Klassifikator
    if C.num_rois > 1:
        if len(pos_samples) < C.num_rois//2:
            selected_pos_samples = pos_samples.tolist()
        else:
            selected_pos_samples = np.random.choice(pos_samples, C.num_rois//2, replace=False).tolist()
        try:
            selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist()
        except:
            selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()
        sel_samples = selected_pos_samples + selected_neg_samples
    else:
        # in the extreme case where num_rois = 1, we pick a random pos or neg sample
        selected_pos_samples = pos_samples.tolist()
        selected_neg_samples = neg_samples.tolist()
        if np.random.randint(0, 2): 
            sel_samples = random.choice(neg_samples)
        else:
            sel_samples = random.choice(pos_samples) 
    return sel_samples


class RPNSequence(Sequence):
    def __init__(self, all_img_data, class_count, C, img_length_calc_function, backend, mode='train'):
        self.all_img_data = all_img_data
        self.class_count = class_count
        self.C = C
        self.img_length_calc_function = img_length_calc_function
        self.backend = backend
        self.mode = mode
        self.batchsize = 1        
        
    def __len__(self):
        return len(self.all_img_data)

    def __getitem__(self,idx):


        if self.mode == 'train':
            np.random.shuffle(self.all_img_data)

        img_data = self.all_img_data[idx*self.batchsize:(idx+1)*self.batch_size]
        try:
            #print('Thread:{} and image: {}'.format(threading.current_thread(), img_data['filepath']))

            # read in image, and optionally add augmentation

            if self.mode == 'train':
                img_data_aug, x_img = data_augment.augment(img_data, C, augment=True)
            else:
                img_data_aug, x_img = data_augment.augment(img_data, C, augment=False)

            (width, height) = (img_data_aug['width'], img_data_aug['height'])
            (rows, cols, _) = x_img.shape

            assert cols == width
            assert rows == height

            # get image dimensions for resizing
            (resized_width, resized_height) = get_new_img_size(width, height, C.im_size)

            # resize the image so that smalles side is length = 600px
            x_img = cv2.resize(x_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)


            y_rpn_cls, y_rpn_regr = calc_rpn(self.C, img_data_aug, width, height, resized_width, resized_height, self.img_length_calc_function)


            # Zero-center by mean pixel, and preprocess image

            x_img = x_img[:,:, (2, 1, 0)]  # BGR -> RGB
            x_img = x_img.astype(np.float32)
            x_img[:, :, 0] -= C.img_channel_mean[0]
            x_img[:, :, 1] -= C.img_channel_mean[1]
            x_img[:, :, 2] -= C.img_channel_mean[2]
            x_img /= C.img_scaling_factor

            x_img = np.transpose(x_img, (2, 0, 1))
            x_img = np.expand_dims(x_img, axis=0)

            y_rpn_regr[:, y_rpn_regr.shape[1]//2:, :, :] *= C.std_scaling

            if self.backend == 'tf':
                x_img = np.transpose(x_img, (0, 2, 3, 1))
                y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1))
                y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1))

            if self.mode == 'pred':
                return np.copy(x_img)
            else:
                return np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug

        except Exception as e:
            print(e)
            

class DetSequence(Sequence):
    def __init__(self, all_img_data, model_rpn, graph, class_count, C, img_length_calc_function, backend, mode='train'):
        self.all_img_data = all_img_data
        self.class_count = class_count
        self.C = C
        self.img_length_calc_function = img_length_calc_function
        self.backend = backend
        self.mode = mode
        self.batchsize = 1
        self.model_rpn = model_rpn
        self.graph = graph

    def __len__(self):
        return len(self.all_img_data)

    def __getitem__(self,idx):
        with graph.as_default():
            model_rpn._make_predict_function()
            
        if self.mode == 'train':
            np.random.shuffle(self.all_img_data)
            
        img_data = self.all_img_data[idx*self.batchsize:(idx+1)*self.batch_size]
        try:
            #print('Thread:{} and image: {}'.format(threading.current_thread(), img_data['filepath']))


            # read in image, and optionally add augmentation
            if self.mode == 'train':
                img_data_aug, x_img = data_augment.augment(img_data, C, augment=True)
            else:
                img_data_aug, x_img = data_augment.augment(img_data, C, augment=False)

            (width, height) = (img_data_aug['width'], img_data_aug['height'])
            (rows, cols, _) = x_img.shape

            
            assert cols == width
            assert rows == height
            
            # get image dimensions for resizing
            (resized_width, resized_height) = get_new_img_size(width, height, C.im_size)
            # resize the image so that smalles side is length = 600px
            x_img = cv2.resize(x_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)

            # Zero-center by mean pixel, and preprocess image

            x_img = x_img[:,:, (2, 1, 0)]  # BGR -> RGB
            x_img = x_img.astype(np.float32)
            x_img[:, :, 0] -= C.img_channel_mean[0]
            x_img[:, :, 1] -= C.img_channel_mean[1]
            x_img[:, :, 2] -= C.img_channel_mean[2]
            x_img /= C.img_scaling_factor

            x_img = np.transpose(x_img, (2, 0, 1))
            x_img = np.expand_dims(x_img, axis=0)
            
            if self.backend == 'tf':
                x_img = np.transpose(x_img, (0, 2, 3, 1))

            #rpn predictions
            with graph.as_default():
                P_rpn = model_rpn.predict(x_img)

            #rpn predictions umformen zu RoI
            R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes=300)

            #X2: RoIs mit Koordinaten (x1, y1, w, h)
            #Y1: Ground truth Klassenlabel der RoIs [0,0,...,0,1,0,0]. Array mit .shape (1, num_rois, num_classes)
            #Y2: Ground truth regression targets der RoIs ohne die Background Klasse:
            #    y_class_regr_label und y_class_regr_coords in einem Array mit .shape = (1, num_rois, (4*num_classes-1)+(4*num_classes-1))
            #IouS: for debugging only
            # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
            X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data_aug, C, C.class_mapping)

            #wenn keine RoI gefunden wurde
            if X2 is None:
                return None
            
            selected_rois_train = select_rois_for_detection(Y1, C)

            return [x_img, X2[:, selected_rois_train, :]], [Y1[:, selected_rois_train, :], Y2[:, selected_rois_train, :]]
            
        except Exception as e:
            print(e)

                
                
try:

    
    parser = OptionParser()
    
    parser.add_option("-p", "--path", dest="train_path", help="Path to training data.", default="~/VOCdevkit/")
    parser.add_option("-n", "--num_rois", type="int", dest="num_rois", help="Number of RoIs to process at once.", default=16)
    parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='vgg16')
    parser.add_option("--hf", dest="horizontal_flips", help="Augment with horizontal flips in training. (Default=false).", action="store_true", default=False)
    parser.add_option("--vf", dest="vertical_flips", help="Augment with vertical flips in training. (Default=false).", action="store_true", default=False)
    parser.add_option("--rot", "--rot_90", dest="rot_90", help="Augment with 90 degree rotations in training. (Default=false).",
    				  action="store_true", default=False)
    parser.add_option("--num_epochs", type="int", dest="num_epochs", help="Number of epochs.", default=2000)
    parser.add_option("--config_filename", dest="config_filename", help=
    				"Location to store all the metadata related to the training (to be used when testing).",
    				default="config.pickle")
    parser.add_option("--output_model_path", dest="output_model_path", help="Output path for model.", default='./train_results/')
    parser.add_option("--input_weight_path", dest="input_weight_path", help="Input path for weights. If not specified, will try to load default weights provided by keras.")
    parser.add_option("--model_name", dest="model_name", help="Output name of model weights.", default='model_frcnn.hdf5')
    parser.add_option("--resume_train", dest="resume_train", help="Lade Zustand des fortzufuehrenden Trainings.",
                      action="store_true", default=False)

    (options, args) = parser.parse_args()
    
    
    
    
    C = config.Config()
    
    # pass the settings from the command line, and persist them in the config object
    # augmented training
    C.use_horizontal_flips = bool(options.horizontal_flips)
    C.use_vertical_flips = bool(options.vertical_flips)
    C.rot_90 = bool(options.rot_90)
    
    # Speicherpfad des trainierten Modells
    model_path = options.output_model_path
    model_name = options.model_name
    
    #batch size fuer den Detektor
    C.num_rois = int(options.num_rois)
    
    if options.network == 'vgg16_small':
        from netze import vgg16_small as nn
        C.network = 'vgg16_small'
    elif options.network == 'vgg16_medium':
        from netze import vgg16_medium as nn
        C.network = 'vgg16_medium'
    elif options.network == 'vgg16':
        from netze import vgg16 as nn
        C.network = 'vgg16'
    else:
        print('Not a valid model')
        raise ValueError
    

     
    #liesst Annotationfiles
    #   all_imgs: ground_truth Bilddaten
    #   classes_count: Anzahl jeder einzelnen Objektklasse
    #   class_mapping: Mapped jede Objektklasse auf eine Zahl (0-19)
    all_imgs, classes_count, class_mapping = get_data(options.train_path)
    
    #fuegt background klasse hinzu
    if 'bg' not in classes_count:
        classes_count['bg'] = 0
        class_mapping['bg'] = len(class_mapping)
    
    #persist class_mapping in config
    C.class_mapping = class_mapping
    
    print('Training images per class:')
    pprint.pprint(classes_count)
    print('Num classes (including bg) = {}'.format(len(classes_count)))
    
    #name of pickled config file
    config_output_filename = model_path + options.config_filename
    with open(config_output_filename, 'wb') as config_f:
    	pickle.dump(C,config_f)
    	print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_output_filename))


    #falls ein Training abgebrochen wurde und weitergefuehrt werden soll wird der Zustand einiger Variablen wiederhergestellt
    #train_seed wird wiederhergestellt um sicherzustellen, dass auf den selben Bildern traniert und validiert wird
    #
    resume_training_path = os.path.join(model_path, 'train_zustand.pickle')
    if options.resume_train:
        with open(resume_training_path, 'rb') as resume_train_file:
            train_seed = pickle.load(resume_train_file)
            incr_valsteps_after_epochs = pickle.load(resume_train_file)
            validation_length = pickle.load(resume_train_file)
            times_increased = pickle.load(resume_train_file)
            patience = pickle.load(resume_train_file)
            wait = pickle.load(resume_train_file)
            min_delta = pickle.load(resume_train_file)
            rpn_history = pickle.load(resume_train_file)
            classifier_history = pickle.load(resume_train_file)
            best_loss = pickle.load(resume_train_file)
            lr_patience = pickle.load(resume_train_file)
            lr_epsilon =pickle.load(resume_train_file)
            lr_reduce_factor = pickle.load(resume_train_file)
            best_rpn_val_loss = pickle.load(resume_train_file)
            lr_rpn_wait = pickle.load(resume_train_file)
            best_det_val_loss = pickle.load(resume_train_file)
            lr_det_wait = pickle.load(resume_train_file)
    else:
        train_seed = random.random()
        incr_valsteps_after_epochs = 4 #erhoehe validation steps, nach x Epochen in denen der Validation Fehler sich nicht gebessert hat
        validation_length = 300
        times_increased = 0
        patience = 20       #early stopping
        wait = 0            #early stopping
        min_delta = 0.005   #early stopping
        rpn_history = []
        classifier_history = []
        best_loss = np.Inf
        lr_patience = 10            #Learning rate reducer
        lr_epsilon = 1e-4           #Learning rate reducer
        lr_reduce_factor= 0.4       #Learning rate reducer
        best_rpn_val_loss = np.Inf  #Learning rate reducer
        lr_rpn_wait = 0             #Learning rate reducer
        best_det_val_loss = np.Inf  #Learning rate reducer
        lr_det_wait = 0             #Learning rate reducer
        
    random.seed(train_seed)
    
    #teile den trainval Datensatz in Trainings- und Validationdatensatz
    trainval_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
    random.shuffle(trainval_imgs)
    num_train_imgs = int((len(trainval_imgs)/100.)*80) #benutze 80 Prozent als Trainingsdaten und den Rest als Validation
    train_imgs = trainval_imgs[:num_train_imgs]
    val_imgs = trainval_imgs[num_train_imgs:]
    print('Num train samples {}'.format(len(train_imgs)))
    print('Num val samples {}'.format(len(val_imgs)))
    
    
    #Netz-Eingabetensoren
    input_shape_img = (None, None, 3) #width*height*colorchannel
    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(None, 4)) #center_x,center_y,width,height
    
    #definiert den Graphen der BaseLayer
    shared_layers = nn.nn_base(img_input, trainable=True)
    
    #baut auf dem Graphen der BaseLayer den RPN Graphen auf
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn = nn.rpn(shared_layers, num_anchors, trainable=True)
    
    
    #definiert den Graphen vom Klassifikator 
    classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count), trainable=True)
    
    #Instantiierung der Modelle
    model_classifier = Model([img_input, roi_input], classifier)
    model_rpn = Model(img_input, rpn[:2])
    # this is a model that holds both the RPN and the classifier, used to load/save weights for the models
    model_all = Model([img_input, roi_input], rpn + classifier)


    # check if weight path was passed via command line
    if options.input_weight_path:
        C.base_net_weights = options.input_weight_path
        try:
            print('loading weights from {}'.format(C.base_net_weights))
            model_rpn.load_weights(C.base_net_weights, by_name=True)
            model_classifier.load_weights(C.base_net_weights, by_name=True)
        except:
            print('Model weights konnten nicht geladen werden.')

    #Modelle kompilieren
    model_rpn.compile(optimizer=Adam(lr=0.00001), loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])
    model_classifier.compile(optimizer=Adam(lr=0.00001), loss=[losses.class_loss_cls, losses.class_loss_regr(len(classes_count)-1)], metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
    model_all.compile(optimizer='sgd', loss='mae')
    model_all.summary()
    

    graph = K.get_session().graph
    
    #
    data_gen_train_rpn = RPNSequence(train_imgs, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='train')
    data_gen_val_rpn = RPNSequence(val_imgs, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='val')
    data_gen_cls_train = DetSequence(train_imgs, model_rpn, graph, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='train')
    data_gen_cls_val = DetSequence(val_imgs, model_rpn, graph, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='train')
    
    
    epoch_length = 1000
    num_epochs = int(options.num_epochs)

    for epoch_num in range(num_epochs):
        print('Trainings Epoche {}/{}'.format(len(rpn_history)+1,num_epochs))
        start_time = time.time()
        
        #Um die Trainingszeit gering zu halten wird mit mit wenigen Validationsteps begonnen und erhoeht wenn
        # das Training anfaengt zu stagnieren um dann Fluktuationen im Validationsfehler zu reduzieren.
        if wait%incr_valsteps_after_epochs==0 and wait/incr_valsteps_after_epochs==times_increased+1:
            times_increased += 1
            validation_length = min(validation_length*2, int(len(val_imgs)/2))
            print('Vergroessere Validationsteps auf {}'.format(validation_length))

        #Trainiere RPN und Classifier im Wechsel fuer je eine Epoche solang EarlyStopping das Training nicht beendet hat
        if wait < patience:
            rpn_hist = model_rpn.fit_generator(generator=data_gen_train_rpn, steps_per_epoch=epoch_length, epochs=1, verbose=1, validation_data=data_gen_val_rpn, validation_steps=validation_length, use_multiprocessing=True, workers=4)
            rpn_history.append(rpn_hist.history)

            det_hist = model_classifier.fit_generator(generator=data_gen_cls_train, steps_per_epoch=epoch_length, epochs=1, verbose=1, validation_data=data_gen_cls_val, validation_steps=validation_length, use_multiprocessing=True, workers=4)
            classifier_history.append(det_hist.history)
        else:
            print('Training wurde beendet durch early stopping nach {} Epochen.'.format(len(rpn_history)+1))
            break

        #speichere Plots aller Losses des Modells
        save_plots_from_history(rpn_history, classifier_history, model_path, len(classes_count))
        
        
        curr_rpn_val_loss = rpn_hist.history['val_loss'][0]
        curr_det_val_loss = det_hist.history['val_loss'][0]
        #Reduzier Learningrate vom model_rpn wenn der val_loss sich um lr_patience Epochen nicht signifikant gebessert hat
        if curr_rpn_val_loss < best_rpn_val_loss - lr_epsilon:
            best_rpn_val_loss = curr_rpn_val_loss
            lr_rpn_wait = 0
        else:
            if lr_rpn_wait >= lr_patience:
                old_lr = float(K.get_value(model_rpn.optimizer.lr))
                if old_lr > 0:
                    new_rpn_lr = old_lr * lr_reduce_factor
                    new_rpn_lr = max(new_rpn_lr, 0)
                    K.set_value(model_rpn.optimizer.lr, new_rpn_lr)
                    print('Reduziere LearningRate vom RPN auf {}.'.format(new_rpn_lr))
                    lr_rpn_wait = 0
            lr_rpn_wait += 1
            
        #Reduzier Learningrate vom model_rpn wenn der val_loss sich um lr_patience Epochen nicht signifikant gebessert hat
        if curr_det_val_loss < best_det_val_loss - lr_epsilon:
            best_det_val_loss = curr_det_val_loss
            lr_det_wait = 0
        else:
            if lr_det_wait >= lr_patience:
                old_lr = float(K.get_value(model_classifier.optimizer.lr))
                if old_lr > 0:
                    new_det_lr = old_lr * lr_reduce_factor
                    new_det_lr = max(new_det_lr, 0)
                    K.set_value(model_classifier.optimizer.lr, new_det_lr)
                    print('Reduziere LearningRate vom Detektor auf {}.'.format(new_det_lr))
                    lr_det_wait = 0
            lr_det_wait += 1
        
        #Wenn Validationloss sich verbessert, dann speichere neues bestes Modell
        curr_val_loss = curr_rpn_val_loss + curr_det_val_loss
        if curr_val_loss < best_loss - min_delta:
            wait = 0
            print('Total validation loss decreased from {} to {}, saving new best weights'.format(best_loss,curr_val_loss))
            best_loss = curr_val_loss
            model_all.save_weights(os.path.join(model_path, 'best_' + model_name))
        else:
            wait += 1
        model_all.save_weights(os.path.join(model_path, model_name))
        
        print('Epoch took: {}'.format(time.time() - start_time))
        
        
        #speichere aktuellen Zustand einiger Variablen um bei Abbruch und fortgefuehrtem Training den Zustand wiederherstellen zu koennen        
        with open(resume_training_path, 'wb') as resume_train_file:
            pickle.dump(train_seed, resume_train_file)
            pickle.dump(incr_valsteps_after_epochs, resume_train_file)
            pickle.dump(validation_length, resume_train_file)
            pickle.dump(times_increased, resume_train_file)
            pickle.dump(patience, resume_train_file)
            pickle.dump(wait, resume_train_file)
            pickle.dump(min_delta, resume_train_file)
            pickle.dump(rpn_history, resume_train_file)
            pickle.dump(classifier_history, resume_train_file)
            pickle.dump(best_loss, resume_train_file)
            pickle.dump(lr_patience, resume_train_file)
            pickle.dump(lr_epsilon, resume_train_file)
            pickle.dump(lr_reduce_factor, resume_train_file)
            pickle.dump(best_rpn_val_loss, resume_train_file)
            pickle.dump(lr_rpn_wait, resume_train_file)
            pickle.dump(best_det_val_loss, resume_train_file)
            pickle.dump(lr_det_wait, resume_train_file)


except Exception:
    print(traceback.format_exc())
