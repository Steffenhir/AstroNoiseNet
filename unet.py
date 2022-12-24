import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L
"""
Architecture from generator Starnet V1 (https://github.com/nekitmm/starnet/)
"""

def unet(window_size, input_channels):
    
    layers = []
    
    filters = [64, 128, 256, 512, 512, 512, 512, 512, 512, 512, 512, 512, 256, 128, 64]
    #filters = [int(1/2*filter) for filter in filters]
    
    input = L.Input(shape=(window_size, window_size, input_channels), name = "gen_input_image")
    
    # layer 0
    convolved = L.Conv2D(filters[0], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(input)
    layers.append(convolved)
        
    # layer 1
    rectified = L.LeakyReLU(alpha = 0.2)(layers[-1])
    convolved = L.Conv2D(filters[1], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
    normalized = L.BatchNormalization()(convolved, training = True)
    layers.append(normalized)
        
    # layer 2
    rectified = L.LeakyReLU(alpha = 0.2)(layers[-1])
    convolved = L.Conv2D(filters[2], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
    normalized = L.BatchNormalization()(convolved, training = True)
    layers.append(normalized)
        
    # layer 3
    rectified = L.LeakyReLU(alpha = 0.2)(layers[-1])
    convolved = L.Conv2D(filters[3], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
    normalized = L.BatchNormalization()(convolved, training = True)
    layers.append(normalized)
        
    # layer 4
    rectified = L.LeakyReLU(alpha = 0.2)(layers[-1])
    convolved = L.Conv2D(filters[4], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
    normalized = L.BatchNormalization()(convolved, training = True)
    layers.append(normalized)
        
    # layer 5
    rectified = L.LeakyReLU(alpha = 0.2)(layers[-1])
    convolved = L.Conv2D(filters[5], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
    normalized = L.BatchNormalization()(convolved, training = True)
    layers.append(normalized)
    
    # layer 6
    rectified = L.LeakyReLU(alpha = 0.2)(layers[-1])
    convolved = L.Conv2D(filters[6], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
    normalized = L.BatchNormalization()(convolved, training = True)
    layers.append(normalized)
    
    # layer 7
    rectified = L.LeakyReLU(alpha = 0.2)(layers[-1])
    convolved = L.Conv2D(filters[7], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
    normalized = L.BatchNormalization()(convolved, training = True)
    layers.append(normalized)
    
    # layer 8
    rectified = L.ReLU()(layers[-1])
    deconvolved = L.Conv2DTranspose(filters[8], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
    normalized = L.BatchNormalization()(deconvolved, training = True)
    layers.append(normalized)
        
    # layer 9
    concatenated = tf.concat([layers[-1], layers[6]], axis = 3)
    rectified = L.ReLU()(concatenated)
    deconvolved = L.Conv2DTranspose(filters[9], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
    normalized = L.BatchNormalization()(deconvolved, training = True)
    layers.append(normalized)
    
    # layer 10
    concatenated = tf.concat([layers[-1], layers[5]], axis = 3)
    rectified = L.ReLU()(concatenated)
    deconvolved = L.Conv2DTranspose(filters[10], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
    normalized = L.BatchNormalization()(deconvolved, training = True)
    layers.append(normalized)
        
    # layer 11
    concatenated = tf.concat([layers[-1], layers[4]], axis = 3)
    rectified = L.ReLU()(concatenated)
    deconvolved = L.Conv2DTranspose(filters[11], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
    normalized = L.BatchNormalization()(deconvolved, training = True)
    layers.append(normalized)
        
    # layer 12
    concatenated = tf.concat([layers[-1], layers[3]], axis = 3)
    rectified = L.ReLU()(concatenated)
    deconvolved = L.Conv2DTranspose(filters[12], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
    normalized = L.BatchNormalization()(deconvolved, training = True)
    layers.append(normalized)
        
    # layer 13
    concatenated = tf.concat([layers[-1], layers[2]], axis = 3)
    rectified = L.ReLU()(concatenated)
    deconvolved = L.Conv2DTranspose(filters[13], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
    normalized = L.BatchNormalization()(deconvolved, training = True)
    layers.append(normalized)
        
    # layer 14
    concatenated = tf.concat([layers[-1], layers[1]], axis = 3)
    rectified = L.ReLU()(concatenated)
    deconvolved = L.Conv2DTranspose(filters[14], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
    normalized = L.BatchNormalization()(deconvolved, training = True)
    layers.append(normalized)
        
    # layer 15
    concatenated = tf.concat([layers[-1], layers[0]], axis = 3)
    rectified = L.ReLU()(concatenated)
    deconvolved = L.Conv2DTranspose(input_channels, kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
    rectified = L.ReLU()(deconvolved)
    output = tf.math.subtract(input, rectified)
    
    return K.Model(inputs = input, outputs = output, name = "generator")