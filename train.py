import random
import pickle
import pprint
import numpy as np
from optparse import OptionParser

from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.optimizers import SGD
from keras.utils import generic_utils

from keras_frcnn import config, data_generators
from keras_frcnn.pascal_voc_parser import get_data
from keras_frcnn import losses as losses

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
parser.add_option("--output_weight_path", dest="output_weight_path", help="Output path for weights.", default='./model_frcnn.hdf5')
parser.add_option("--input_weight_path", dest="input_weight_path", help="Input path for weights. If not specified, will try to load default weights provided by keras.")

(options, args) = parser.parse_args()

C = config.Config()

# pass the settings from the command line, and persist them in the config object
# augmented training
C.use_horizontal_flips = bool(options.horizontal_flips)
C.use_vertical_flips = bool(options.vertical_flips)
C.rot_90 = bool(options.rot_90)

# Speicherpfad des trainierten Modells
C.model_path = options.output_weight_path

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
config_output_filename = options.config_filename
with open(config_output_filename, 'wb') as config_f:
	pickle.dump(C,config_f)
	print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_output_filename))
 
random.shuffle(all_imgs)

#teile all_imgs in Trainings- und Validationdatensatz
train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
val_imgs = [s for s in all_imgs if s['imageset'] == 'test']
print('Num train samples {}'.format(len(train_imgs)))
print('Num val samples {}'.format(len(val_imgs)))

#
data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='train')
data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='val')

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
model_rpn.compile(optimizer=SGD(lr=0.001), loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])
model_classifier.compile(optimizer=SGD(lr=0.001), loss=[losses.class_loss_cls, losses.class_loss_regr(len(classes_count)-1)], metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
model_all.compile(optimizer='sgd', loss='mae')
model_all.summary()

epoch_length = 1000
num_epochs = int(options.num_epochs)
iter_num = 0

best_loss = np.Inf
losses = np.zeros((epoch_length, 5))


print('Starting training')
for epoch_num in range(num_epochs):

    progbar = generic_utils.Progbar(epoch_length)
    print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))
    
    while True: #add early stopping

        try:
            X, Y, img_data = next(data_gen_train)
            
#            loss_rpn = model_rpn.train_on_batch(X, Y)
            
#            P_rpn = model_rpn.predict_on_batch(X)
            
            
        except Exception as e:
            print('Exception: {}'.format(e))
            continue
        