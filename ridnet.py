import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L


class EAM(tf.keras.layers.Layer):
  def __init__(self,**kwargs):
    super().__init__(**kwargs)
    
    self.conv1 = L.Conv2D(64, (3,3), dilation_rate=1,padding='same',activation='relu')
    self.conv2 = L.Conv2D(64, (3,3), dilation_rate=2,padding='same',activation='relu') 

    self.conv3 = L.Conv2D(64, (3,3), dilation_rate=3,padding='same',activation='relu')
    self.conv4 = L.Conv2D(64, (3,3), dilation_rate=4,padding='same',activation='relu')

    self.conv5 = L.Conv2D(64, (3,3),padding='same',activation='relu')

    self.conv6 = L.Conv2D(64, (3,3),padding='same',activation='relu')
    self.conv7 = L.Conv2D(64, (3,3),padding='same')

    self.conv8 = L.Conv2D(64, (3,3),padding='same',activation='relu')
    self.conv9 = L.Conv2D(64, (3,3),padding='same',activation='relu')
    self.conv10 = L.Conv2D(64, (1,1),padding='same')

    self.gap = L.GlobalAveragePooling2D()

    self.conv11 = L.Conv2D(64, (3,3),padding='same',activation='relu')
    self.conv12 = L.Conv2D(64, (3,3),padding='same',activation='sigmoid')

  def call(self,input):
    conv1 = self.conv1(input)
    conv1 = self.conv2(conv1)

    conv2 = self.conv3(input)
    conv2 = self.conv4(conv2)

    concat = L.concatenate([conv1,conv2])
    conv3 = self.conv5(concat)
    add1 = L.Add()([input,conv3])

    conv4 = self.conv6(add1)
    conv4 = self.conv7(conv4)
    add2 = L.Add()([conv4,add1])
    add2 = L.Activation('relu')(add2)

    conv5 = self.conv8(add2)
    conv5 = self.conv9(conv5)
    conv5 = self.conv10(conv5)
    add3 = L.Add()([add2,conv5])
    add3 = L.Activation('relu')(add3)

    gap = self.gap(add3)
    gap = L.Reshape((1,1,64))(gap)
    conv6 = self.conv11(gap)
    conv6 = self.conv12(conv6)
    
    mul = L.Multiply()([conv6, add3])
    out = L.Add()([input,mul]) # This is not included in the reference code
    return out
  






def ridnet(window_size, input_channels):
    
    input = L.Input(shape=(window_size, window_size, input_channels), name = "gen_input_image")
    
    conv1 = L.Conv2D(64, (3,3),padding='same')(input)
    eam1 = EAM()(conv1)
    eam2 = EAM()(eam1)
    eam3 = EAM()(eam2)
    eam4 = EAM()(eam3) 
    conv2 = L.Conv2D(3, (3,3),padding='same')(eam4)
    out = L.Add()([conv2,input])

    return K.Model(input,out)
    