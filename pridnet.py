import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L

"""
Architecture of Pyramid Real Image Denoising Network (https://arxiv.org/abs/1908.00273)
"""

def lrelu(x):
    return tf.maximum(x * 0.2, x)



def upsample_and_concat(x1, x2, output_channels, in_channels, scope_name, trainable=True):

    pool_size = 2
    #deconv_filter = tf.Variable(tf.random.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    #deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2) , strides=[1, pool_size, pool_size, 1], name=scope_name)
    
    deconv = L.Conv2DTranspose(output_channels, pool_size, strides=pool_size,padding="SAME")(x1)
    deconv_output =  tf.concat([deconv, x2],3)
    deconv_output.set_shape([None, None, None, output_channels*2])

    return deconv_output


def unet(input):
    conv1 = L.Conv2D(32, kernel_size = 3, strides = (1, 1), padding = "SAME", kernel_initializer = tf.initializers.GlorotUniform())(input)
    conv1 = L.LeakyReLU(alpha = 0.2)(conv1)
    conv1 = L.Conv2D(32, kernel_size = 3, strides = (1, 1), padding = "SAME", kernel_initializer = tf.initializers.GlorotUniform())(conv1)
    conv1 = L.LeakyReLU(alpha = 0.2)(conv1)
    conv1 = L.Conv2D(32, kernel_size = 3, strides = (1, 1), padding = "SAME", kernel_initializer = tf.initializers.GlorotUniform())(conv1)
    conv1 = L.LeakyReLU(alpha = 0.2)(conv1)
    conv1 = L.Conv2D(32, kernel_size = 3, strides = (1, 1), padding = "SAME", kernel_initializer = tf.initializers.GlorotUniform())(conv1)
    conv1 = L.LeakyReLU(alpha = 0.2)(conv1)
    pool1 = L.MaxPooling2D(pool_size = (2,2), strides = 2, padding='SAME')(conv1)
    

    conv2 = L.Conv2D(64, kernel_size = 3, strides = (1, 1), padding = "SAME", kernel_initializer = tf.initializers.GlorotUniform())(pool1)
    conv2 = L.LeakyReLU(alpha = 0.2)(conv2)
    conv2 = L.Conv2D(64, kernel_size = 3, strides = (1, 1), padding = "SAME", kernel_initializer = tf.initializers.GlorotUniform())(conv2)
    conv2 = L.LeakyReLU(alpha = 0.2)(conv2)
    conv2 = L.Conv2D(64, kernel_size = 3, strides = (1, 1), padding = "SAME", kernel_initializer = tf.initializers.GlorotUniform())(conv2)
    conv2 = L.LeakyReLU(alpha = 0.2)(conv2)
    conv2 = L.Conv2D(64, kernel_size = 3, strides = (1, 1), padding = "SAME", kernel_initializer = tf.initializers.GlorotUniform())(conv2)
    conv2 = L.LeakyReLU(alpha = 0.2)(conv2)
    pool2 = L.MaxPooling2D(pool_size = (2,2), strides = 2, padding='SAME')(conv2)

    conv3 = L.Conv2D(128, kernel_size = 3, strides = (1, 1), padding = "SAME", kernel_initializer = tf.initializers.GlorotUniform())(pool2)
    conv3 = L.LeakyReLU(alpha = 0.2)(conv3)
    conv3 = L.Conv2D(128, kernel_size = 3, strides = (1, 1), padding = "SAME", kernel_initializer = tf.initializers.GlorotUniform())(conv3)
    conv3 = L.LeakyReLU(alpha = 0.2)(conv3)
    conv3 = L.Conv2D(128, kernel_size = 3, strides = (1, 1), padding = "SAME", kernel_initializer = tf.initializers.GlorotUniform())(conv3)
    conv3 = L.LeakyReLU(alpha = 0.2)(conv3)
    conv3 = L.Conv2D(128, kernel_size = 3, strides = (1, 1), padding = "SAME", kernel_initializer = tf.initializers.GlorotUniform())(conv3)
    conv3 = L.LeakyReLU(alpha = 0.2)(conv3)
    pool3 = L.MaxPooling2D(pool_size = (2,2), strides = 2, padding='SAME')(conv3)

    conv4 = L.Conv2D(256, kernel_size = 3, strides = (1, 1), padding = "SAME", kernel_initializer = tf.initializers.GlorotUniform())(pool3)
    conv4 = L.LeakyReLU(alpha = 0.2)(conv4)
    conv4 = L.Conv2D(256, kernel_size = 3, strides = (1, 1), padding = "SAME", kernel_initializer = tf.initializers.GlorotUniform())(conv4)
    conv4 = L.LeakyReLU(alpha = 0.2)(conv4)
    conv4 = L.Conv2D(256, kernel_size = 3, strides = (1, 1), padding = "SAME", kernel_initializer = tf.initializers.GlorotUniform())(conv4)
    conv4 = L.LeakyReLU(alpha = 0.2)(conv4)
    conv4 = L.Conv2D(256, kernel_size = 3, strides = (1, 1), padding = "SAME", kernel_initializer = tf.initializers.GlorotUniform())(conv4)
    conv4 = L.LeakyReLU(alpha = 0.2)(conv4)
    pool4 = L.MaxPooling2D(pool_size = (2,2), strides = 2, padding='SAME')(conv4)

    conv5 = L.Conv2D(512, kernel_size = 3, strides = (1, 1), padding = "SAME", kernel_initializer = tf.initializers.GlorotUniform())(pool4)
    conv5 = L.LeakyReLU(alpha = 0.2)(conv5)
    conv5 = L.Conv2D(512, kernel_size = 3, strides = (1, 1), padding = "SAME", kernel_initializer = tf.initializers.GlorotUniform())(conv5)
    conv5 = L.LeakyReLU(alpha = 0.2)(conv5)
    conv5 = L.Conv2D(512, kernel_size = 3, strides = (1, 1), padding = "SAME", kernel_initializer = tf.initializers.GlorotUniform())(conv5)
    conv5 = L.LeakyReLU(alpha = 0.2)(conv5)
    conv5 = L.Conv2D(512, kernel_size = 3, strides = (1, 1), padding = "SAME", kernel_initializer = tf.initializers.GlorotUniform())(conv5)
    conv5 = L.LeakyReLU(alpha = 0.2)(conv5)

    up6 = upsample_and_concat(conv5, conv4, 256, 512, "up6")
    conv6 = L.Conv2D(256, kernel_size = 3, strides = (1, 1), padding = "SAME", kernel_initializer = tf.initializers.GlorotUniform())(up6)
    conv6 = L.LeakyReLU(alpha = 0.2)(conv6)
    conv6 = L.Conv2D(256, kernel_size = 3, strides = (1, 1), padding = "SAME", kernel_initializer = tf.initializers.GlorotUniform())(conv6)
    conv6 = L.LeakyReLU(alpha = 0.2)(conv6)
    conv6 = L.Conv2D(256, kernel_size = 3, strides = (1, 1), padding = "SAME", kernel_initializer = tf.initializers.GlorotUniform())(conv6)
    conv6 = L.LeakyReLU(alpha = 0.2)(conv6)

    up7 = upsample_and_concat(conv6, conv3, 128, 256, "up7")
    conv7 = L.Conv2D(128, kernel_size = 3, strides = (1, 1), padding = "SAME", kernel_initializer = tf.initializers.GlorotUniform())(up7)
    conv7 = L.LeakyReLU(alpha = 0.2)(conv7)
    conv7 = L.Conv2D(128, kernel_size = 3, strides = (1, 1), padding = "SAME", kernel_initializer = tf.initializers.GlorotUniform())(conv7)
    conv7 = L.LeakyReLU(alpha = 0.2)(conv7)
    conv7 = L.Conv2D(128, kernel_size = 3, strides = (1, 1), padding = "SAME", kernel_initializer = tf.initializers.GlorotUniform())(conv7)
    conv7 = L.LeakyReLU(alpha = 0.2)(conv7)

    up8 = upsample_and_concat(conv7, conv2, 64, 128, "up8")
    conv8 = L.Conv2D(64, kernel_size = 3, strides = (1, 1), padding = "SAME", kernel_initializer = tf.initializers.GlorotUniform())(up8)
    conv8 = L.LeakyReLU(alpha = 0.2)(conv8)
    conv8 = L.Conv2D(64, kernel_size = 3, strides = (1, 1), padding = "SAME", kernel_initializer = tf.initializers.GlorotUniform())(conv8)
    conv8 = L.LeakyReLU(alpha = 0.2)(conv8)
    conv8 = L.Conv2D(64, kernel_size = 3, strides = (1, 1), padding = "SAME", kernel_initializer = tf.initializers.GlorotUniform())(conv8)
    conv8 = L.LeakyReLU(alpha = 0.2)(conv8)

    up9 = upsample_and_concat(conv8, conv1, 32, 64, "up9")
    conv9 = L.Conv2D(32, kernel_size = 3, strides = (1, 1), padding = "SAME", kernel_initializer = tf.initializers.GlorotUniform())(up9)
    conv9 = L.LeakyReLU(alpha = 0.2)(conv9)
    conv9 = L.Conv2D(32, kernel_size = 3, strides = (1, 1), padding = "SAME", kernel_initializer = tf.initializers.GlorotUniform())(conv9)
    conv9 = L.LeakyReLU(alpha = 0.2)(conv9)
    conv9 = L.Conv2D(32, kernel_size = 3, strides = (1, 1), padding = "SAME", kernel_initializer = tf.initializers.GlorotUniform())(conv9)
    conv9 = L.LeakyReLU(alpha = 0.2)(conv9)
    
    conv10 = L.Conv2D(3, kernel_size = 1, strides = (1, 1), padding = "SAME", kernel_initializer = tf.initializers.GlorotUniform())(conv9)

    #out = tf.depth_to_space(conv10, 2)
    return conv10


def feature_encoding(input):
    conv1 = L.Conv2D(32, kernel_size = 3, strides = (1, 1), padding = "SAME", kernel_initializer = tf.initializers.GlorotUniform())(input)
    activ1 = L.LeakyReLU(alpha = 0.2)(conv1)
    
    conv2 = L.Conv2D(32, kernel_size = 3, strides = (1, 1), padding = "SAME", kernel_initializer = tf.initializers.GlorotUniform())(activ1)
    activ2 = L.LeakyReLU(alpha = 0.2)(conv2)
    
    conv3 = L.Conv2D(32, kernel_size = 3, strides = (1, 1), padding = "SAME", kernel_initializer = tf.initializers.GlorotUniform())(activ2)
    activ3 = L.LeakyReLU(alpha = 0.2)(conv3)
    
    conv4 = L.Conv2D(32, kernel_size = 3, strides = (1, 1), padding = "SAME", kernel_initializer = tf.initializers.GlorotUniform())(activ3)
    activ4 = L.LeakyReLU(alpha = 0.2)(conv4)
    

    
    squeeze = squeeze_excitation_layer(activ4, 32, 6)
    output = L.Conv2D(3, kernel_size = 3, strides = (1, 1), padding = "SAME", kernel_initializer = tf.initializers.GlorotUniform())(squeeze)
    output = L.LeakyReLU(alpha = 0.2)(output)

    return output


def squeeze_excitation_layer(input_x, out_dim, middle):
    squeeze = L.GlobalAveragePooling2D()(input_x)
    excitation = L.Dense(use_bias=True, units=middle)(squeeze)
    excitation = tf.nn.relu(excitation)
    excitation = L.Dense(use_bias=True, units=out_dim)(excitation)
    excitation = tf.nn.sigmoid(excitation)
    excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
    scale = input_x * excitation
    return scale


def avg_pool(feature_map):
    ksize = [[1, 1, 1, 1], [1, 2, 2, 1], [1, 4, 4, 1], [1, 8, 8, 1], [1, 16, 16, 1]]
    pool1 = tf.nn.avg_pool(feature_map, ksize=ksize[0], strides=ksize[0], padding='VALID')
    pool2 = tf.nn.avg_pool(feature_map, ksize=ksize[1], strides=ksize[1], padding='VALID')
    pool3 = tf.nn.avg_pool(feature_map, ksize=ksize[2], strides=ksize[2], padding='VALID')
    pool4 = tf.nn.avg_pool(feature_map, ksize=ksize[3], strides=ksize[3], padding='VALID')
    pool5 = tf.nn.avg_pool(feature_map, ksize=ksize[4], strides=ksize[4], padding='VALID')

    return pool1, pool2, pool3, pool4, pool5


def all_unet(pool1, pool2, pool3, pool4, pool5):
    unet1 = unet(pool1)
    unet2 = unet(pool2)
    unet3 = unet(pool3)
    unet4 = unet(pool4)
    unet5 = unet(pool5)

    return unet1, unet2, unet3, unet4, unet5


def resize_all_image(unet1, unet2, unet3, unet4, unet5):
    resize1 = tf.image.resize(images=unet1, size=[tf.shape(unet1, out_type=tf.int32)[1],tf.shape(unet1, out_type=tf.int32)[2]], method=tf.image.ResizeMethod.BILINEAR)
    resize2 = tf.image.resize(images=unet2, size=[tf.shape(unet1, out_type=tf.int32)[1],tf.shape(unet1, out_type=tf.int32)[2]], method=tf.image.ResizeMethod.BILINEAR)
    resize3 = tf.image.resize(images=unet3, size=[tf.shape(unet1, out_type=tf.int32)[1],tf.shape(unet1, out_type=tf.int32)[2]], method=tf.image.ResizeMethod.BILINEAR)
    resize4 = tf.image.resize(images=unet4, size=[tf.shape(unet1, out_type=tf.int32)[1],tf.shape(unet1, out_type=tf.int32)[2]], method=tf.image.ResizeMethod.BILINEAR)
    resize5 = tf.image.resize(images=unet5, size=[tf.shape(unet1, out_type=tf.int32)[1],tf.shape(unet1, out_type=tf.int32)[2]], method=tf.image.ResizeMethod.BILINEAR)

    return resize1, resize2, resize3, resize4, resize5


def to_clean_image(feature_map, resize1, resize2, resize3, resize4, resize5):
    concat = tf.concat([feature_map, resize1, resize2, resize3, resize4, resize5], 3)

    sk_conv1 = L.Conv2D(21, kernel_size = 3, strides = (1, 1), padding = "SAME", kernel_initializer = tf.initializers.GlorotUniform())(concat)
    sk_conv1 = L.LeakyReLU(alpha = 0.2)(sk_conv1)
    sk_conv2 = L.Conv2D(21, kernel_size = 5, strides = (1, 1), padding = "SAME", kernel_initializer = tf.initializers.GlorotUniform())(concat)
    sk_conv2 = L.LeakyReLU(alpha = 0.2)(sk_conv2)
    sk_conv3 = L.Conv2D(21, kernel_size = 7, strides = (1, 1), padding = "SAME", kernel_initializer = tf.initializers.GlorotUniform())(concat)
    sk_conv3 = L.LeakyReLU(alpha = 0.2)(sk_conv3)

    sk_out = selective_kernel_layer(sk_conv1, sk_conv2, sk_conv3, 2, 21)

    output = L.Conv2D(3, kernel_size = 3, strides = (1, 1), padding = "SAME", kernel_initializer = tf.initializers.GlorotUniform())(sk_out)
    #output = L.Conv2D(3, kernel_size = 3, strides = (1, 1), padding = "SAME", kernel_initializer = tf.initializers.GlorotUniform())(concat)

    return output



def selective_kernel_layer(sk_conv1, sk_conv2, sk_conv3, middle, out_dim):
    sum_u = sk_conv1 + sk_conv2 + sk_conv3
    squeeze = L.GlobalAveragePooling2D()(sum_u)
    squeeze = tf.reshape(squeeze, [-1, 1, 1, out_dim])
    z = L.Dense(use_bias=True, units=middle)(squeeze)
    z = tf.nn.relu(z)
    a1 = L.Dense(use_bias=True, units=out_dim)(z)
    a2 = L.Dense(use_bias=True, units=out_dim)(z)
    a3 = L.Dense(use_bias=True, units=out_dim)(z)

    before_softmax = tf.concat([a1, a2, a3], 1)
    after_softmax = tf.nn.softmax(before_softmax)
    a1 = after_softmax[:, 0, :, :]
    a1 = tf.reshape(a1, [-1, 1, 1, out_dim])
    a2 = after_softmax[:, 1, :, :]
    a2 = tf.reshape(a2, [-1, 1, 1, out_dim])
    a3 = after_softmax[:, 2, :, :]
    a3 = tf.reshape(a3, [-1, 1, 1, out_dim])

    select_1 = sk_conv1 * a1
    select_2 = sk_conv2 * a2
    select_3 = sk_conv3 * a3

    out = select_1 + select_2 + select_3

    return out


def pridnet(window_size, input_channels):
    
    input = L.Input(shape=(window_size, window_size, input_channels), name = "gen_input_image")
    
    feature_map = feature_encoding(input)
    feature_map_2 = tf.concat([input, feature_map], 3)
    pool1, pool2, pool3, pool4, pool5 = avg_pool(feature_map_2)
    unet1, unet2, unet3, unet4, unet5 = all_unet(pool1, pool2, pool3, pool4, pool5)
    resize1, resize2, resize3, resize4, resize5 = resize_all_image(unet1, unet2, unet3, unet4, unet5)
    out_image = to_clean_image(feature_map_2, resize1, resize2, resize3, resize4, resize5)
    
    out_image = tf.subtract(input, out_image)
    return K.Model(inputs = input, outputs = out_image, name = "generator")