import pickle
import pprint
import numpy as np
from optparse import OptionParser

from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.optimizers import SGD

from keras_frcnn import config
from keras_frcnn.pascal_voc_parser import get_data

parser = OptionParser()

parser.add_option("-p", "--path", dest="train_path", help="Path to training data.", default="~/VOCdevkit/")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois", help="Number of RoIs to process at once.", default=32)
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='resnet50') #change default to mine
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

config_output_filename = options.config_filename