from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from keras.layers import TimeDistributed
from keras_frcnn.RoiPoolingConv import RoiPoolingConv


def get_img_output_length(width, height):
    '''
    Abhaengig von der Stridegroesse der Basis Layer wird die height und width der resultierenden Feature Map nach
    Anwendung der Basis Layer auf das Bild zurueckgegeben.
    '''
    def get_output_length(input_length):
        return input_length//16

    return get_output_length(width), get_output_length(height)
    
def nn_base(img_input, trainable=False):
    '''
    Definition der Basis Layer.
    '''
    # Block 1
    x = Conv2D(64, (3, 3), kernel_initializer='he_normal', activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), kernel_initializer='he_normal', activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), kernel_initializer='he_normal', activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), kernel_initializer='he_normal', activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), kernel_initializer='he_normal', activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), kernel_initializer='he_normal', activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), kernel_initializer='he_normal', activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), kernel_initializer='he_normal', activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), kernel_initializer='he_normal', activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), kernel_initializer='he_normal', activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), kernel_initializer='he_normal', activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), kernel_initializer='he_normal', activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), kernel_initializer='he_normal', activation='relu', padding='same', name='block5_conv3')(x)

    return x
    
def rpn(base_layers, num_anchors, trainable=False):
    '''
    Definiert das Region Proposal Netzwerk. Auf den base_layers wird ein weiterer feature extractor hinzugefuegt auf dem
    dann Objectness Scores und BBox Regression stattfinden.
    '''
    
    x = Conv2D(512, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu', name='rpn_conv1')(base_layers)

    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]


def classifier(base_layers, input_rois, num_rois, nb_classes = 21, trainable=False):
    '''
    Definiert das Klassifikator Netzwerk. Der RoI Pooling Layer baut auf den base_layers und den input_rois auf. Darauf
    folgen TimeDistributed Layer (Keras Layer Wrapper), mit der die RoIs (Anzahl gegeben durch num_rois) parallel durch
    fully-connected Layer bis zur Klassifikation und Regression durchpropagiert werden.
    '''
    #pooling_regions = 7 resultiert in 7*7 pools
    pooling_regions = 7

    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])

    #Die Eingabe (out_roi_pool) in die TimeDistributed Layer ist: (1, num_rois, channels, pooling_regions, pooling_regions)
    out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
    out = TimeDistributed(Dense(2048, kernel_initializer='he_normal', activation='relu', name='fc1'))(out)
    out = TimeDistributed(Dropout(0.5))(out)
    out = TimeDistributed(Dense(2048, kernel_initializer='he_normal', activation='relu', name='fc2'))(out)
    out = TimeDistributed(Dropout(0.5))(out)

    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)

    return [out_class, out_regr]
