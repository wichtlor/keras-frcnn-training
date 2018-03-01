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
from keras.utils import generic_utils
from keras.callbacks import EarlyStopping

from keras_frcnn import config
from keras_frcnn.pascal_voc_parser import get_data
from keras_frcnn import losses as losses

from visualization.plots import save_plots_from_history
from module import data_generators
from module.earlystopping import MyEarlyStopping

try:

    
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
    parser.add_option("--seed", dest="use_seed", help="Benutze den random.seed eines vorangegangenen Trainings. seed.pickle muss im Ordner des zu trainierenden Modells liegen.",
                      action="store_true", default=False)

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
    elif options.network == 'vgg16':
    	from netze import mynet as nn
    	C.network = 'mynet'
    elif options.network == 'mynet_small':
    	from netze import mynet_small as nn
    	C.network = 'mynet_small'
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
    config_output_filename = C.model_path + options.config_filename
    with open(config_output_filename, 'wb') as config_f:
    	pickle.dump(C,config_f)
    	print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_output_filename))


    #falls ein Training abgebrochen wurde und weitergefuehrt werden soll oder man immer auf den selben Bildern Trainieren und Validieren moechte
    #kann der seed festgelegt werden
    seed_path = os.path.join(C.model_path, 'seed.pickle')
    if options.use_seed:
        with open(seed_path, 'rb') as random_seed:
            train_seed = pickle.load(random_seed)
    else:
        train_seed = random.random()
        with open(seed_path, 'wb') as random_seed:
            pickle.dump(train_seed, random_seed)
    
    random.seed(train_seed)
    
    #teile den trainval Datensatz in Trainings- und Validationdatensatz
    trainval_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
    random.shuffle(trainval_imgs)
    num_train_imgs = int((len(trainval_imgs)/100.)*80) #benutze 80 Prozent als Trainingsdaten und den Rest als Validation
    train_imgs = trainval_imgs[:num_train_imgs]
    val_imgs = trainval_imgs[num_train_imgs:]
    print('Num train samples {}'.format(len(train_imgs)))
    print('Num val samples {}'.format(len(val_imgs)))
    
    
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
    model_classifier = Model([img_input, roi_input], classifier)
    model_rpn = Model(img_input, rpn[:2])
    # this is a model that holds both the RPN and the classifier, used to load/save weights for the models
    model_all = Model([img_input, roi_input], rpn + classifier)
    
    best_loss = np.Inf
    rpn_history = []
    classifier_history = []
    
    # check if weight path was passed via command line
    if options.input_weight_path:
        C.base_net_weights = options.input_weight_path
        try:
            print('loading weights from {}'.format(C.base_net_weights))
            model_rpn.load_weights(C.base_net_weights, by_name=True)
            model_classifier.load_weights(C.base_net_weights, by_name=True)
        except:
            print('Model weights konnten nicht geladen werden.')

        if 'losses.pickle' in os.listdir(options.input_weight_path.rsplit(os.sep,1)[0]):
            print('Losses von vorangegangenem Training werden geladen.')
            pickled_losses_path = options.input_weight_path.rsplit(os.sep,1)[0] + os.sep + 'losses.pickle'
            with open(pickled_losses_path, 'rb') as read_loss:
                rpn_history = pickle.load(read_loss)
                classifier_history = pickle.load(read_loss)
                best_loss = pickle.load(read_loss)

    #Modelle kompilieren
    model_rpn.compile(optimizer=Adam(lr=0.00001), loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])
    model_classifier.compile(optimizer=Adam(lr=0.00001), loss=[losses.class_loss_cls, losses.class_loss_regr(len(classes_count)-1)], metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
    model_all.compile(optimizer='sgd', loss='mae')
    model_all.summary()
    
    
    rpn_accuracy_rpn_monitor_train = []
    rpn_accuracy_for_epoch_train = []
    
    graph = K.get_session().graph
    
    #
    data_gen_train_rpn = data_generators.get_anchor_gt(train_imgs, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='train')
    data_gen_val_rpn = data_generators.get_anchor_gt(val_imgs, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='val')
    data_gen_cls_train = data_generators.get_classifier_gt(train_imgs, model_rpn, graph, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='train')
    data_gen_cls_val = data_generators.get_classifier_gt(val_imgs, model_rpn, graph, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='train')
    
    
    epoch_length = 5
    validation_length = 2
    num_epochs = int(options.num_epochs)
    

    train_losses = np.zeros((epoch_length, 5))
    epoch_mean_losses = np.zeros((num_epochs, 10))
    
    rpn_es = MyEarlyStopping(monitor='val_loss', min_delta=0, patience=3) #todo: bei fortgefahrenem Training resettet patience
    det_es = MyEarlyStopping(monitor='val_loss', min_delta=0, patience=3)
    rpn_stopped_epoch = 0
    det_stopped_epoch = 0
    
    for epoch_num in range(num_epochs):
        print('Trainings Epoche {}/{}'.format(len(rpn_history)+1,num_epochs))
        start_time = time.time()
        
        #Trainiere RPN und Classifier im Wechsel fuer je eine Epoche solang die EarlyStopping callbacks das Training nicht beendet haben
        if rpn_stopped_epoch==0:
            rpn_hist = model_rpn.fit_generator(generator=data_gen_train_rpn, steps_per_epoch=epoch_length, epochs=1, callbacks=[rpn_es], verbose=1, validation_data=data_gen_val_rpn, validation_steps=validation_length, use_multiprocessing=False, workers=2)
            rpn_history.append(rpn_hist.history)
            
            if rpn_es.stopped_epoch!=0:
                print('RPN STOP rpn_es.stopped_epoch={}'.format(rpn_es.stopped_epoch))
                rpn_stopped_epoch = epoch_num
        else:
            rpn_history.append(rpn_hist.history)
        
        if det_stopped_epoch==0:
            det_hist = model_classifier.fit_generator(generator=data_gen_cls_train, steps_per_epoch=epoch_length, epochs=1, callbacks=[det_es], verbose=1, validation_data=data_gen_cls_val, validation_steps=validation_length, use_multiprocessing=False, workers=2)
            classifier_history.append(det_hist.history)
            if det_es.stopped_epoch!=0:
                print('DET STOP det_es.stopped_epoch={}'.format(det_es.stopped_epoch))
                det_stopped_epoch = epoch_num
        else:
            classifier_history.append(det_hist.history)
        
        #pickle losses um auch nach abgebrochenem und weitergefuehrtem Training vollstaendige Lossplots zu bekommen
        with open(os.path.join(C.model_path, 'losses.pickle'), 'wb') as pickle_loss:
            pickle.dump(rpn_history, pickle_loss)
            pickle.dump(classifier_history, pickle_loss)
            pickle.dump(best_loss, pickle_loss)
            
        #speichere Plots aller Losses des Modells
        curr_val_loss = save_plots_from_history(rpn_history, classifier_history, C.model_path, len(classes_count))
        
        #Wenn Validationloss sich verbessert, dann speichere weights
        if curr_val_loss < best_loss:
            print('Total validation loss decreased from {} to {}, saving weights'.format(best_loss,curr_val_loss))
            best_loss = curr_val_loss
            model_all.save_weights(os.path.join(C.model_path, model_name))
            
        print('Epoch took: {}'.format(time.time() - start_time))
        
        if rpn_stopped_epoch!=0 and det_stopped_epoch!=0:
            print('Training wurde beendet durch early stopping nach {} RPN Epochen und {} Detektor Epochen'.format(rpn_stopped_epoch,det_stopped_epoch))
            break
        
except Exception:
    print(traceback.format_exc())
