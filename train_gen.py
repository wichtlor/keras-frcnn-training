import random
import pickle
import pprint
import time
import numpy as np
from optparse import OptionParser

from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.optimizers import SGD, Adam
from keras.utils import generic_utils

from keras_frcnn import config
from keras_frcnn.pascal_voc_parser import get_data
from keras_frcnn import losses as losses
from keras_frcnn import roi_helpers as roi_helpers

from visualization.plots import save_plots
from module import data_generators



parser = OptionParser()

parser.add_option("-p", "--path", dest="train_path", help="Path to training data.", default="~/VOCdevkit/")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois", help="Number of RoIs to process at once.", default=32)
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='mynet') #change default to mine
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
(options, args) = parser.parse_args()




C = config.Config()

# pass the settings from the command line, and persist them in the config object
# augmented training
C.use_horizontal_flips = bool(options.horizontal_flips)
C.use_vertical_flips = bool(options.vertical_flips)
C.rot_90 = bool(options.rot_90)

# Speicherpfad des trainierten Modells
C.model_path = options.output_model_path
model_name = options.model_name

#batch size fuer den Detektor
C.num_rois = int(options.num_rois)

if options.network == 'vgg':
	C.network = 'vgg'
	from keras_frcnn import vgg as nn
elif options.network == 'resnet50':
	from keras_frcnn import resnet as nn
	C.network = 'resnet50'
elif options.network == 'mynet':
	from netze import mynet as nn
	C.network = 'mynet'
elif options.network == 'mynet_small':
	from netze import mynet_small as nn
	C.network = 'mynet_small'
else:
	print('Not a valid model')
	raise ValueError

# check if weight path was passed via command line
if options.input_weight_path:
    C.base_net_weights = options.input_weight_path
 
 
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
config_output_filename = C.model_path + options.config_filename
with open(config_output_filename, 'wb') as config_f:
	pickle.dump(C,config_f)
	print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_output_filename))
 
#random.shuffle(all_imgs)#bilder werden auch noch im data generator geshuffled...???

#teile all_imgs in Trainings- und Validationdatensatz
train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
val_imgs = [s for s in all_imgs if s['imageset'] == 'test']
print('Num train samples {}'.format(len(train_imgs)))
print('Num val samples {}'.format(len(val_imgs)))

#
data_gen_train_rpn = data_generators.get_anchor_gt(train_imgs, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='train')
data_gen_val_rpn = data_generators.get_anchor_gt(val_imgs, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='val')

#Netz-Eingabetensor
input_shape_img = (None, None, 3) #width*height*colorchannel
img_input = Input(shape=input_shape_img)

roi_input = Input(shape=(None, 4)) #center_x,center_y,width,height

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = nn.rpn(shared_layers, num_anchors, trainable=True)

#RoI Klassifikator 
classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count), trainable=True)

#Instantiierung der Modelle
model_rpn = Model(img_input, rpn)
model_classifier = Model([img_input, roi_input], classifier)
# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
model_all = Model([img_input, roi_input], rpn + classifier)



#==============================================================================
# try:
# 	print('loading weights from {}'.format(C.base_net_weights))
# 	model_rpn.load_weights(C.base_net_weights, by_name=True)
# 	model_classifier.load_weights(C.base_net_weights, by_name=True)
# except:
# 	print('Could not load pretrained model weights. Weights can be found in the keras application folder \
# 		https://github.com/fchollet/keras/tree/master/keras/applications')
# 
#==============================================================================



#Modelle kompilieren
model_rpn.compile(optimizer=Adam(lr=0.0005), loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])
model_classifier.compile(optimizer=Adam(lr=0.0005), loss=[losses.class_loss_cls, losses.class_loss_regr(len(classes_count)-1)], metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
model_all.compile(optimizer='sgd', loss='mae')
model_all.summary()

epoch_length = 1000
validation_length = 200
num_epochs = int(options.num_epochs)
iter_num = 0

best_loss = np.Inf
train_losses = np.zeros((epoch_length, 5))
epoch_mean_losses = np.zeros((num_epochs, 10))

rpn_accuracy_rpn_monitor_train = []
rpn_accuracy_for_epoch_train = []
start_time = time.time()





model_rpn.fit_generator(generator=data_gen_train_rpn, steps_per_epoch=30, epochs=5, verbose=1, validation_data=data_gen_val_rpn, validation_steps=5, workers=4, use_multiprocessing=True)

data_gen_cls_train = data_generators.get_classifier_gt(train_imgs, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='train', model_rpn)
data_gen_cls_val = data_generators.get_classifier_gt(val_imgs, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='train', model_rpn)

model_classifier.fit_generator(generator=data_gen_cls_train, steps_per_epoch=30, epochs=5, verbose=1, validation_data=data_gen_cls_val, validation_steps=5, workers=4, use_multiprocessing=True)


#==============================================================================
# 
# print('Starting training')
# for epoch_num in range(num_epochs):
# 
#     progbar = generic_utils.Progbar(epoch_length)
#     print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))
#     
#     while True: #add early stopping for rpn and det
# 
#         try:
#             
#             #???
#             if len(rpn_accuracy_rpn_monitor_train) == epoch_length and C.verbose:
#                 mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor_train))/len(rpn_accuracy_rpn_monitor_train)
#                 rpn_accuracy_rpn_monitor_train = []
#                 print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, epoch_length))
#                 if mean_overlapping_bboxes == 0:
#                     print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')
# 
# 
#             #X:
#             #Y:
#             #img_data:
#             X, Y, img_data = next(data_gen_train)
#             
#             #train rpn und get rpn_train_loss
#             rpn_train_loss = model_rpn.train_on_batch(X, Y)
#             
#             #rpn predictions
#             P_rpn = model_rpn.predict_on_batch(X)
#             
#             #rpn predictions umformen zu RoI
#             R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes=300)
#             
#             #X2: RoIs mit Koordinaten (x1, y1, w, h)
#             #Y1: Ground truth Klassenlabel der RoIs [0,0,...,0,1,0,0]. Array mit .shape (1, num_rois, num_classes)
#             #Y2: Ground truth regression targets der RoIs ohne die Background Klasse:
#             #    y_class_regr_label und y_class_regr_coords in einem Array mit .shape = (1, num_rois, (4*num_classes-1)+(4*num_classes-1))
#             #IouS: for debugging only
#             # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
#             X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)            
#             
#             #wenn keine RoI gefunden wurde
#             if X2 is None:
#                 rpn_accuracy_rpn_monitor_train.append(0)
#                 rpn_accuracy_for_epoch_train.append(0)
#                 continue
#             
#             selected_rois_train = select_rois_for_detection(Y1)
#     
#             #train detektor und get det_train_loss
#             det_train_loss = model_classifier.train_on_batch([X, X2[:, selected_rois_train, :]], [Y1[:, selected_rois_train, :], Y2[:, selected_rois_train, :]])
# 
#             #speicher rpn losses
#             train_losses[iter_num, 0] = rpn_train_loss[1]
#             train_losses[iter_num, 1] = rpn_train_loss[2]
#             #speicher detektor losses
#             train_losses[iter_num, 2] = det_train_loss[1]
#             train_losses[iter_num, 3] = det_train_loss[2]
#             train_losses[iter_num, 4] = det_train_loss[3]
#             
#             iter_num += 1
# 
#             progbar.update(iter_num, [('rpn_cls', np.mean(train_losses[:iter_num, 0])), ('rpn_regr', np.mean(train_losses[:iter_num, 1])),
# 									  ('detector_cls', np.mean(train_losses[:iter_num, 2])), ('detector_regr', np.mean(train_losses[:iter_num, 3]))])
# 
#             #Ende der Epoche
#             if iter_num == epoch_length:
# 
#                 epoch_mean_losses[epoch_num,0] = np.mean(train_losses[:, 0])
#                 epoch_mean_losses[epoch_num,1] = np.mean(train_losses[:, 1])
#                 epoch_mean_losses[epoch_num,2] = np.mean(train_losses[:, 2])
#                 epoch_mean_losses[epoch_num,3] = np.mean(train_losses[:, 3])
#                 epoch_mean_losses[epoch_num,4] = np.mean(train_losses[:, 4])
#                 
#                 mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch_train)) / len(rpn_accuracy_for_epoch_train) #??
#                 rpn_accuracy_for_epoch_train = [] #??
#                 
# 
#                 print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
#                 print('Classifier accuracy for bounding boxes from RPN: {}'.format(epoch_mean_losses[epoch_num,4]))
#                 print('Loss RPN classifier: {}'.format(epoch_mean_losses[epoch_num,0]))
#                 print('Loss RPN regression: {}'.format(epoch_mean_losses[epoch_num,1]))
#                 print('Loss Detector classifier: {}'.format(epoch_mean_losses[epoch_num,2]))
#                 print('Loss Detector regression: {}'.format(epoch_mean_losses[epoch_num,3]))
#                 print('Elapsed time: {}'.format(time.time() - start_time))
#                 
#                 curr_loss = epoch_mean_losses[epoch_num,0] + epoch_mean_losses[epoch_num,1] + epoch_mean_losses[epoch_num,2] + epoch_mean_losses[epoch_num,3]
#                 print('Current Training Loss is: {}'.format(curr_loss))
#                 
#                 start_time = time.time()                
#                 iter_num = 0
#                 
#                 
#                 #validation
#                 val_on_num_pictures = 200 #Anzahl der Batches (Bilder in diesem Fall) auf denen validiert werden soll
#                 val_losses = np.zeros((val_on_num_pictures, 5))
#                 print('validation start')
#                 progbar2 = generic_utils.Progbar(val_on_num_pictures)
#                 while True:
# 
#                     X_val, Y_val, img_data_val = next(data_gen_val)
# 
#                     #get rpn_val_loss
#                     rpn_val_loss = model_rpn.test_on_batch(X_val, Y_val)
# 
#                     #get rpn predictions fuer detektor validation
#                     predict_rpn_val = model_rpn.predict_on_batch(X_val)
#                     
#                     R = roi_helpers.rpn_to_roi(predict_rpn_val[0], predict_rpn_val[1], C, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes=300)
#                     
#                     X2_val, Y1_val, Y2_val, IouS_val = roi_helpers.calc_iou(R, img_data_val, C, class_mapping)
# 
#                     #wenn keine RoI gefunden wurde
#                     if X2_val is None:
#                         continue
#                     
#                     selected_rois_val = select_rois_for_detection(Y1_val)
#                     #get det_val_loss
#                     det_val_loss = model_classifier.test_on_batch([X_val, X2_val[:, selected_rois_val, :]], [Y1_val[:, selected_rois_val, :], Y2_val[:, selected_rois_val, :]])
# 
#                     #speicher rpn losses
#                     val_losses[iter_num, 0] = rpn_val_loss[1]
#                     val_losses[iter_num, 1] = rpn_val_loss[2]
#                     #speicher detektor losses
#                     val_losses[iter_num, 2] = det_val_loss[1]
#                     val_losses[iter_num, 3] = det_val_loss[2]
#                     val_losses[iter_num, 4] = det_val_loss[3]
#                     
#                     iter_num += 1
#                     progbar2.update(iter_num, [('rpn_cls_val', np.mean(val_losses[:iter_num, 0])), ('rpn_regr_val', np.mean(val_losses[:iter_num, 1])),
# 									  ('detector_cls_val', np.mean(val_losses[:iter_num, 2])), ('detector_regr_val', np.mean(val_losses[:iter_num, 3]))])
# 
#                     #Ende der Validation
#                     if iter_num == val_on_num_pictures:
#                         
#                         epoch_mean_losses[epoch_num,5] = np.mean(val_losses[:, 0])
#                         epoch_mean_losses[epoch_num,6] = np.mean(val_losses[:, 1])
#                         epoch_mean_losses[epoch_num,7] = np.mean(val_losses[:, 2])
#                         epoch_mean_losses[epoch_num,8] = np.mean(val_losses[:, 3])
#                         epoch_mean_losses[epoch_num,9] = np.mean(val_losses[:, 4])
# 
#                         
#                         print('Validation Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
#                         print('Validation Classifier accuracy for bounding boxes from RPN: {}'.format(epoch_mean_losses[epoch_num,9]))
#                         print('Validation Loss RPN classifier: {}'.format(epoch_mean_losses[epoch_num,5]))
#                         print('Validation Loss RPN regression: {}'.format(epoch_mean_losses[epoch_num,6]))
#                         print('Validation Loss Detector classifier: {}'.format(epoch_mean_losses[epoch_num,7]))
#                         print('Validation Loss Detector regression: {}'.format(epoch_mean_losses[epoch_num,8]))
#                         print('Elapsed time: {}'.format(time.time() - start_time))
#                         
#                         curr_val_loss = epoch_mean_losses[epoch_num,5] + epoch_mean_losses[epoch_num,6] + epoch_mean_losses[epoch_num,7] + epoch_mean_losses[epoch_num,8]
#                         print('Current Validation Loss is: {}'.format(curr_val_loss))
# 
#                         if curr_val_loss < best_loss: #anpassen an validation loss
#                             print('Total validation loss decreased from {} to {}, saving weights'.format(best_loss,curr_val_loss))
#                             best_loss = curr_val_loss
#                             model_all.save_weights(C.model_path + model_name)
#                         start_time = time.time()
#                         save_plots(epoch_mean_losses, epoch_num+1, C.model_path)
#                         print('Saving plots took: {}'.format(time.time() - start_time))
#                         start_time = time.time()
#                         iter_num = 0
#                         break
#                 
#                 break
#             
#             
#         except Exception as e:
#             print('Exception: {}'.format(e))
#             continue
# print('Training complete, exiting.')
#==============================================================================
