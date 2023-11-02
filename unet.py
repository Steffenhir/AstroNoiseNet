import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L
"""
Architecture from generator Starnet V1 (https://github.com/nekitmm/starnet/)
"""

def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result


def upsample(filters, size, apply_dropout=False, apply_nn=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  
  if apply_nn:
      result.add(tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear"))
      
      result.add(
          tf.keras.layers.Conv2D(filters, size, strides=1, padding='same',
                                 kernel_initializer=initializer, use_bias=False))
  else:
      result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))
 
  
  
  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result


def unet(window_size, input_channels):
  inputs = L.Input(shape=(window_size, window_size, input_channels), name = "gen_input_image")
  
  down_stack = [
      downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
      downsample(128, 4),  # (batch_size, 64, 64, 128)
      downsample(256, 4),  # (batch_size, 32, 32, 256)
      downsample(512, 4),  # (batch_size, 16, 16, 512)
      downsample(512, 4),  # (batch_size, 8, 8, 512)
      downsample(512, 4),  # (batch_size, 4, 4, 512)
      downsample(512, 4),  # (batch_size, 2, 2, 512)
      downsample(512, 4),  # (batch_size, 1, 1, 512)
    ]
    
  up_stack = [
      upsample(512, 4, apply_dropout=True,apply_nn=False),  # (batch_size, 2, 2, 1024)
      upsample(512, 4, apply_dropout=True,apply_nn=False),  # (batch_size, 4, 4, 1024)
      upsample(512, 4, apply_dropout=True,apply_nn=False),  # (batch_size, 8, 8, 1024)
      upsample(512, 4,apply_nn=False),  # (batch_size, 16, 16, 1024)
      upsample(256, 4,apply_nn=False),  # (batch_size, 32, 32, 512)
      upsample(128, 4,apply_nn=False),  # (batch_size, 64, 64, 256)
      upsample(64, 4,apply_nn=False),  # (batch_size, 128, 128, 128)
    ]

  
  initializer = tf.random_normal_initializer(0., 0.02)
  
  last = tf.keras.layers.Conv2DTranspose(input_channels, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

  
  """
  last = tf.keras.Sequential()
  last.add(tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear"))
  last.add(tf.keras.layers.Conv2D(3, 4, strides=1, padding='same',
                         kernel_initializer=initializer, activation='tanh'))
  """
  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)
  
  x = tf.subtract(inputs, x)

  return tf.keras.Model(inputs=inputs, outputs=x)
