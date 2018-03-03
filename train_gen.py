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

from keras_frcnn import config
from keras_frcnn.pascal_voc_parser import get_data
from keras_frcnn import losses as losses

from visualization.plots import save_plots_from_history
from module import data_generators


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
    
    if options.network == 'mynet_small':
        from netze import mynet_small as nn
        C.network = 'mynet_small'
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
    else:
        train_seed = random.random()
        incr_valsteps_after_epochs = 4 #erhoehe validation steps, nach x Epochen in denen der Validation Fehler sich nicht gebessert hat
        validation_length = 1
        times_increased = 0
        patience = 20
        wait = 0
        min_delta = 0.003
        rpn_history = []
        classifier_history = []
        best_loss = np.Inf
        
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
    
    
    rpn_accuracy_rpn_monitor_train = []
    rpn_accuracy_for_epoch_train = []
    
    graph = K.get_session().graph
    
    #
    data_gen_train_rpn = data_generators.get_anchor_gt(train_imgs, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='train')
    data_gen_val_rpn = data_generators.get_anchor_gt(val_imgs, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='val')
    data_gen_cls_train = data_generators.get_classifier_gt(train_imgs, model_rpn, graph, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='train')
    data_gen_cls_val = data_generators.get_classifier_gt(val_imgs, model_rpn, graph, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='train')
    
    
    epoch_length = 2
    num_epochs = int(options.num_epochs)

    for epoch_num in range(num_epochs):
        print('Trainings Epoche {}/{}'.format(len(rpn_history)+1,num_epochs))
        start_time = time.time()
        
        #Um die Trainingszeit gering zu halten wird mit mit wenigen Validationsteps begonnen und erhoeht wenn
        # das Training anfaengt zu stagnieren um dann Fluktuationen im Validationsfehler zu reduzieren.
        if wait%incr_valsteps_after_epochs==0 and wait/incr_valsteps_after_epochs==times_increased+1:
            times_increased += 1
            validation_length = min(validation_length*2, len(val_imgs))
            print('Vergroessere Validationsteps auf {}'.format(validation_length))

        #Trainiere RPN und Classifier im Wechsel fuer je eine Epoche solang EarlyStopping das Training nicht beendet hat
        if wait < patience:
            rpn_hist = model_rpn.fit_generator(generator=data_gen_train_rpn, steps_per_epoch=epoch_length, epochs=1, verbose=1, validation_data=data_gen_val_rpn, validation_steps=validation_length, use_multiprocessing=False, workers=2)
            rpn_history.append(rpn_hist.history)

            det_hist = model_classifier.fit_generator(generator=data_gen_cls_train, steps_per_epoch=epoch_length, epochs=1, verbose=1, validation_data=data_gen_cls_val, validation_steps=validation_length, use_multiprocessing=False, workers=2)
            classifier_history.append(det_hist.history)
        else:
            print('Training wurde beendet durch early stopping nach {} Epochen.'.format(len(rpn_history)+1))
            break

        #speichere Plots aller Losses des Modells
        curr_val_loss = save_plots_from_history(rpn_history, classifier_history, model_path, len(classes_count))
        
        #Wenn Validationloss sich verbessert, dann speichere neues bestes Modell
        if curr_val_loss < best_loss and best_loss-curr_val_loss > min_delta:
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

except Exception:
    print(traceback.format_exc())
