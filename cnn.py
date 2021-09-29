import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, DepthwiseConv2D, BatchNormalization, Conv2D, LeakyReLU, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.regularizers import l2


def seperable_cnn(dim, output_neurons, output_activation):
    def block(x, n_filters, d_strides):
        # depthwise
        x = DepthwiseConv2D(kernel_size = (3,3), strides = d_strides, padding = 'same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
    
        # pointwise
        x = Conv2D(filters = n_filters, kernel_size = (1,1))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        
        # max-pool
        x = MaxPooling2D(pool_size = (2,2), strides = 2, padding = 'same')(x)
    
        return x
    
    
    input = Input(shape = dim)
    
    k = 16
    x = Conv2D(filters = k, kernel_size = (3,3), strides = 2, padding = 'same')(input)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = block(x, n_filters = 2*k, d_strides = 1)

    x = block(x, n_filters = 4*k, d_strides = 1)    

    x = block(x, n_filters = 8*k, d_strides = 1)    

    x = block(x, n_filters = 16*k, d_strides = 1)
    x = Dropout(0.2)(x)

    x = block(x, n_filters = 32*k, d_strides = 1)
    
    x = GlobalAveragePooling2D()(x)
    
    output = Dense(output_neurons, output_activation)(x)  
    
    model = Model(inputs = input, outputs = output)
    
    return model