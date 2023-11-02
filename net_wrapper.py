from os import listdir
from os.path import isfile, join
import numpy as np

import warnings
warnings.simplefilter('ignore', np.RankWarning)

import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L
import copy
import pickle
import scipy
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb 


from stretch import stretch
from pridnet import pridnet
from unet import unet
from ridnet import ridnet

from IPython import display

class Net():
    def __init__(self, mode:str, window_size:int = 512, stride:int = 256, lr:float = 1e-4, train_folder:str = './train/', batch_size:int = 1,
                 validation_folder:str = "./validation/", validation:bool = False):
        assert mode in ['RGB', 'Greyscale'], "Mode should be either RGB or Greyscale"
        self.mode = mode
        if self.mode == 'RGB': self.input_channels = 3
        else: self.input_channels = 1
        self.window_size = window_size
        self.stride = stride
        self.train_folder = train_folder
        self.validation_folder = validation_folder
        self.validation = validation
        self.batch_size = batch_size
        self.history = {}
        self.val_history = {}
        self.weights = []
        self.lr = lr
        
        self.short = []
        self.long = []   
        
    def __str__(self):
        return "Net instance"
    
        
    def load_training_dataset(self):
        self.weights = []
        short_files = [f for f in listdir(self.train_folder + "/short/") if isfile(join(self.train_folder + "/short/", f))\
                          and f.endswith(".fits")]
        long_files = [f for f in listdir(self.train_folder + "/long/") if isfile(join(self.train_folder + "/long/", f))\
                          and f.endswith(".fits")]
        
        assert len(short_files) == len(long_files), 'Numbers of files in `long` and `short` subfolders should be equal'
        
        assert len(short_files) > 0 and len(long_files) > 0, 'No training data found in {}'.format(self.train_folder)
        
        
        print("Total training images found: {}".format(len(short_files)))
        
        self.short = []
        self.long = []
        
        self.median = []
        self.mad = []
        
        for i in short_files:
            if self.mode == "RGB":
                self.short.append(np.moveaxis(fits.getdata(self.train_folder + "/short/" + i, ext=0), 0, 2))

        for i in long_files:
            self.long.append(np.moveaxis(fits.getdata(self.train_folder + "/long/" + i, ext=0), 0, 2))
        
        for img in self.short:
            median = np.median(img[::4,::4,:], axis=[0,1])
            mad = np.median(np.abs(img[::4,::4,:]-median), axis=[0,1])
            
            self.median.append(median)
            self.mad.append(mad)
        
        
        total_pixels = 0
        
        for i in range(len(short_files)):
            assert self.short[i].shape == self.long[i].shape, 'Image sizes are not equal: {}/short/{} and {}/long/{}'\
                                                                      .format(self.train_folder, short_files[i],\
                                                                      self.train_folder, long_files[i])
            
            total_pixels += self.short[i].shape[0] * self.short[i].shape[1]
            self.weights.append(self.short[i].shape[0] * self.short[i].shape[1])
        
        print("Total size of training images: %.2f MP" % (total_pixels / 1e6))
        
        self.iters_per_epoch = total_pixels // (self.window_size * self.window_size) // 2
        
        self.weights = [i / np.sum(self.weights) for i in self.weights]
        
        print("One epoch is set to %d iterations" % self.iters_per_epoch)
        print("Training dataset has been successfully loaded!")
        
        if self.validation:
            self.load_validation_dataset()

        
    def load_validation_dataset(self):
        val_short_files = [f for f in listdir(self.validation_folder + "/short/") if isfile(join(self.validation_folder + "/short/", f))\
                          and f.endswith(".fits")]
        val_long_files = [f for f in listdir(self.validation_folder + "/long/") if isfile(join(self.validation_folder + "/long/", f))\
                          and f.endswith(".fits")]
        
        assert len(val_short_files) == len(val_long_files), 'Numbers of files in `long` and `short` validation subfolders should be equal'
        
        assert len(val_short_files) > 0 and len(val_long_files) > 0, 'No validation data found in {}'.format(self.validation_train_folder)
        
        for i in range(len(val_short_files)):
            assert(val_short_files[i] == val_long_files[i]), 'Corresponding names of short and long validation files should be equal'
        
        print("Total validation images found: {}".format(len(val_short_files)))
        
        self.val_short = []
        self.val_long = []
        
        self.val_median = []
        self.val_mad = []
        
        for i in val_short_files:
            if self.mode == "RGB":
                self.val_short.append(np.moveaxis(fits.getdata(self.validation_folder + "/short/" + i, ext=0), 0, 2))
                self.val_long.append(np.moveaxis(fits.getdata(self.validation_folder + "/long/" + i, ext=0), 0, 2))
                
                     
            else:
                self.val_short.append(np.moveaxis(np.array([fits.getdata(self.validation_folder + "/short/" + i, ext=0)]), 0, 2))
                self.val_long.append(np.moveaxis(np.array([fits.getdata(self.validation_folder + "/long/" + i, ext=0)]), 0, 2))
        
        
        linked_stretch = True
        
        for image in self.val_short:
            median = []
            mad = []
            
            if linked_stretch:
                for c in range(image.shape[-1]):
                    median.append(np.median(image[:,:,:]))
                    mad.append(np.median(np.abs(image[:,:,:] - median[c])))
            else:
                for c in range(image.shape[-1]):
                    median.append(np.median(image[:,:,c]))
                    mad.append(np.median(np.abs(image[:,:,c] - median[c])))
                
            self.val_median.append(median)
            self.val_mad.append(mad)
            
        
        print("Validation dataset has been successfully loaded!")
 

    def load_model(self, weights = None, history = None):
        self.G = self._generator()
        self.D = self._discriminator()
        
        self.gen_optimizer = tf.optimizers.Adam(self.lr)
        self.dis_optimizer = tf.optimizers.Adam(self.lr / 4)
        
        self.D.build(input_shape = (None, self.window_size, self.window_size, self.input_channels))
        self.G.build(input_shape = (None, self.window_size, self.window_size, self.input_channels))
        

        if weights:
            self.G.load_weights(weights + '_G_' + self.mode + '.h5')
            #self.D.load_weights(weights + '_D_' + self.mode + '.h5')
        if history:
            with open(history + '_' + self.mode + '.pkl', "rb") as h:
                self.history = pickle.load(h)
                
            with open(history + '_val_' + self.mode + '.pkl', "rb") as h:
                self.val_history = pickle.load(h)
  
    def initialize_model(self):
        self.load_model()
    
    def _ramp(self, x):
        return tf.clip_by_value(x, 0, 1)
    
    def linear_fit(self, o, s, clipping):
        for c in range(o.shape[-1]):
            indx_clipped = o[:,:,c].flatten() < clipping
            coeff = np.polyfit(s[:,:,c].flatten()[indx_clipped], o[:,:,c].flatten()[indx_clipped], 1)
            s[:,:,c] = s[:,:,c]*coeff[0] + coeff[1]
            
    def _augmentator(self, o, s, median, mad):

        # flip horizontally
        if np.random.rand() < 0.50:
            o = np.flip(o, axis = 1)
            s = np.flip(s, axis = 1)
        
        # flip vertically
        if np.random.rand() < 0.50:
            o = np.flip(o, axis = 0)
            s = np.flip(s, axis = 0)
        
        # rotate 90, 180 or 270
        if np.random.rand() < 0.75:
            k = int(np.random.rand() * 3 + 1)
            o = np.rot90(o, k, axes = (1, 0))
            s = np.rot90(s, k, axes = (1, 0))
        
        """
        if self.mode == 'RGB':
            
            o_hsv = rgb_to_hsv(o)
            s_hsv = rgb_to_hsv(s)
            
            # tweak hue
            hue = np.random.normal(0,0.2)
            o_hsv[:,:,0] += hue
            s_hsv[:,:,0] += hue
            
            o_hsv[:,:,0] = np.where(o_hsv[:,:,0] < 0, o_hsv[:,:,0] + 1, o_hsv[:,:,0])
            o_hsv[:,:,0] = np.where(o_hsv[:,:,0] > 1, o_hsv[:,:,0] - 1, o_hsv[:,:,0])
            s_hsv[:,:,0] = np.where(s_hsv[:,:,0] < 0, s_hsv[:,:,0] + 1, s_hsv[:,:,0])
            s_hsv[:,:,0] = np.where(s_hsv[:,:,0] > 1, s_hsv[:,:,0] - 1, s_hsv[:,:,0])
        
            # tweak saturation
            sat = np.random.normal(1.25,0.25)
            o_hsv[:,:,1] *= sat
            s_hsv[:,:,1] *= sat           
            
            # tweak value
            val = np.random.normal(0,0.1)
            o_hsv[:,:,2] += val
            s_hsv[:,:,2] += val
            
            o_hsv = np.clip(o_hsv,0,1)
            s_hsv = np.clip(s_hsv,0,1)
            
            o[:,:,:] = hsv_to_rgb(o_hsv)
            s[:,:,:] = hsv_to_rgb(s_hsv)
                    
        else:
            # tweak brightness
            if np.random.rand() < 0.70:
                m = np.min((o, s))
                offset = np.random.rand() * 0.25 - np.random.rand() * m
                o[:, :] = o[:, :] + offset * (1.0 - o[:, :])
                s[:, :] = s[:, :] + offset * (1.0 - s[:, :])
        """
        #o_median = np.median(o ,axis=[0,1])
        #o_mad = np.median(np.abs(o - o_median), axis=[0,1])
        #s_median = np.median(s ,axis=[0,1])
        #s_mad = np.median(np.abs(s - s_median), axis=[0,1])
        
        o = (o - median) / mad * 0.04
        s = (s - median) / mad * 0.04
        self.linear_fit(o, s, 5)
        
        o = np.clip(o, -1.0, 1.0)
        s = np.clip(s, -1.0, 1.0)

        
        return o, s

            
    def _get_sample(self, r, h, w, type:str):
        assert type in ['short', 'long']
        if type == 'short':
            return np.copy(self.short[r][h:h+self.window_size, w:w+self.window_size])
        else:
            return np.copy(self.long[r][h:h+self.window_size, w:w+self.window_size])
        
    def generate_input(self, iterations = 1, augmentation = False):
        for _ in range(iterations):
            o = np.zeros((self.batch_size, self.window_size, self.window_size, self.input_channels), dtype = np.float32)
            s = np.zeros((self.batch_size, self.window_size, self.window_size, self.input_channels), dtype = np.float32)
                
            for i in range(self.batch_size):
                if augmentation:
                    r = int(np.random.choice(range(len(self.short)), 1, p = self.weights))
                    h = np.random.randint(self.short[r].shape[0] - self.window_size)
                    w = np.random.randint(self.short[r].shape[1] - self.window_size)
                    o[i], s[i] = self._augmentator(self._get_sample(r, h, w, type = 'short'),\
                                                   self._get_sample(r, h, w, type = 'long'), self.median[r], self.mad[r])
                else:
                    r = int(np.random.choice(range(len(self.short)), 1, p = self.weights))
                    h = np.random.randint(self.short[r].shape[0] - self.window_size)
                    w = np.random.randint(self.short[r].shape[1] - self.window_size)
                    o[i] = self._get_sample(r, h, w, type = 'short')
                    s[i] = self._get_sample(r, h, w, type = 'long')
        return o, s
        
        
    def train(self, epochs, augmentation = True, plot_progress = False, plot_interval = 50, save_backups = False, warm_up = False):
        assert self.short != [], 'Training dataset was not loaded, use load_training_dataset() first'
        
        for e in range(epochs):
            for i in range(self.iters_per_epoch):
                
                if self.validation and i % 1000 == 0 and i != 0:
                    self.validate()
                
                x, y = self.generate_input(augmentation = augmentation)
                
                if warm_up: y = x
                
                if i % plot_interval == 0 and plot_progress:
                    plt.close()
                    fig, ax = plt.subplots(1, 4, sharex = True, figsize=(16.5, 16.5))
                    if self.mode == 'RGB':
                        ax[0].imshow(x[0]*3 + 0.2,vmin=0,vmax=1)
                        ax[0].set_title('short')
                        ax[1].imshow(self.G(x)[0]*3 + 0.2,vmin=0,vmax=1)
                        ax[1].set_title('output')
                        ax[2].imshow(y[0]*3 + 0.2,vmin=0,vmax=1)
                        ax[2].set_title('long')
                        
                        ax[3].imshow(30 * np.abs(y[0] - self.G(x)[0]))
                        ax[3].set_title('Difference x30')
                    
                    display.clear_output(wait = True)
                    display.display(plt.gcf())
                
                if i > 0:
                    print("\rEpoch: %d. Iteration %d / %d Loss %f L1 Loss %f Gen Loss GAN %f " % (e, i, self.iters_per_epoch, np.mean(self.history['total'][-500:]), np.mean(self.history['gen_L1'][-500:]), np.mean(self.history['gen_loss_GAN'][-500:])), end = '')
                    #print("\rEpoch: %d. Iteration %d / %d L1 Loss %f   " % (e, i, self.iters_per_epoch, self.history['gen_L1'][-1]), end = '')
                else:
                    print("\rEpoch: %d. Iteration %d / %d " % (e, i, self.iters_per_epoch), end = '')
                
                
                with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
                    gen_output = self.G(x)
                    
                    disc_generated_output = self.D([x, gen_output])
                    disc_real_output = self.D([x, y])
                    
                    
                    
                    d = {}
                    
                    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
                    gen_loss_GAN = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
                    d['gen_loss_GAN'] = gen_loss_GAN
                    
                    dis_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output) + loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
                    d['dis_loss'] = dis_loss
                    
                    
                    gen_L1 = tf.reduce_mean(tf.abs(y - gen_output))
                    d['gen_L1'] = gen_L1 * 100
                    
                    
                    #gen_loss = 0.1 * (gen_loss_GAN * 0.1 + gen_p1 * 0.1 + gen_p2 * 10 + gen_p3 * 10 + gen_p4 * 10 + gen_p5 * 10 + gen_p6 * 10 + gen_p7 * 10 + gen_p8 * 10) + gen_L1 * 100
                    #gen_loss = 100 * gen_L1 + 0.02*gen_loss_GAN
                    gen_loss = 100 * gen_L1
                    
                    d['total'] = gen_loss
                    
                    for k in d:
                        if k in self.history.keys():
                            self.history[k].append(d[k])
                        else:
                            self.history[k] = [d[k]]
                    
                    gen_grads = gen_tape.gradient(gen_loss, self.G.trainable_variables)
                    self.gen_optimizer.apply_gradients(zip(gen_grads, self.G.trainable_variables))
                    
                    dis_grads = dis_tape.gradient(dis_loss, self.D.trainable_variables)
                    self.dis_optimizer.apply_gradients(zip(dis_grads, self.D.trainable_variables))
                    
                    
            if save_backups:
                if e % 2 == 0:
                    self.G.save_weights("./Net_backup_G_even.h5")
                    self.D.save_weights("./Net_backup_D_even.h5")
                else:
                    self.G.save_weights("./Net_backup_G_odd.h5")
                    self.D.save_weights("./Net_backup_D_odd.h5")
            
            if plot_progress: plt.close()

    
    def validate(self):
        
        print("Start validation")
        
        val_metrics = {"L1_loss": 0.0, "dis_loss": 0.0, "psnr": 0.0, "SSIM": 0.0}

        
        for i in range(len(self.val_short)):
            h, w, _ = self.val_short[i].shape
            
            ith = h // self.window_size
            itw = w // self.window_size
            
            num_iterations = 0
            
            for x in range(ith):
                for y in range(itw):
                    num_iterations += 1
                    
                    # Slice
                    short = self.val_short[i][x*self.window_size:(x+1)*self.window_size,y*self.window_size:(y+1)*self.window_size]
                    long = self.val_long[i][x*self.window_size:(x+1)*self.window_size,y*self.window_size:(y+1)*self.window_size]
                    
                    self.linear_fit(short, long, 0.95)
                    
                    # Stretch
                    bg = 0.2
                    sigma = 3.0
                    short, long = stretch(short, long, bg, sigma, self.val_median[i], self.val_mad[i])
                    
                    
                    output = self.G(np.expand_dims(short * 2 - 1, axis = 0))[0]
                    output = (output + 1) / 2
                    
                    # Calculate metrics
                    val_metrics["L1_loss"] += tf.reduce_mean(tf.abs(long - output)) * 2 * 100
                    
                    p1_real, p2_real, p3_real, p4_real, p5_real, p6_real, p7_real, p8_real, predict_real = self.D(np.expand_dims(long*2 - 1, axis = 0))
                    p1_fake, p2_fake, p3_fake, p4_fake, p5_fake, p6_fake, p7_fake, p8_fake, predict_fake = self.D(np.expand_dims(output*2 - 1, axis = 0))
                    
                    val_metrics["dis_loss"] += tf.reduce_mean(-(tf.math.log(predict_real + 1E-8) + tf.math.log(1 - predict_fake + 1E-8)))
                    
                    val_metrics["psnr"] += tf.image.psnr(long, output, max_val = 1.0)

                    val_metrics["SSIM"] += tf.image.ssim(long, output, max_val = 1.0)
                    
                    
        
        for metric in val_metrics:
            if metric in self.val_history:
                self.val_history[metric].append(val_metrics[metric] / num_iterations)
            else:
                self.val_history[metric] = [val_metrics[metric] / num_iterations]
                
            print(metric + ": " + str(val_metrics[metric] / num_iterations))
        
        
        print("Finished validation")
                    
                           
    
    def plot_history(self, last = None):
        assert self.history != {}, 'Empty training history, nothing to plot'
        fig, ax = plt.subplots(4, 3, sharex = True, figsize=(16, 14))
        
        keys = list(self.history.keys())
        
        keys = [k for k in keys if k != '']
        
        for i in range(4):
            for j in range(3):
                if last: ax[i][j].plot(self.history[keys[j+3*i]][-last:])
                else: ax[i][j].plot(self.history[keys[j+3*i]])
                ax[i][j].set_title(keys[j+3*i])
                
        
        if self.validation:
        
            fig, ax = plt.subplots(1, 4, sharex = True, figsize=(16, 14))
            
            keys = list(self.val_history.keys())
            
            keys = [k for k in keys if k != '']
            
            for i in range(4):
                if last: ax[i].plot(self.val_history[keys[i]][-last:])
                else: ax[i].plot(self.val_history[keys[i]])
                ax[i].set_title(keys[i])
                
    def save_model(self, weights_filename, history_filename = None):

        self.G.save_weights(weights_filename + '_G_' + self.mode + '.h5')
        self.D.save_weights(weights_filename + '_D_' + self.mode + '.h5')
        if history_filename:
            with open(history_filename + '_' + self.mode + '.pkl', 'wb') as f:
                pickle.dump(self.history, f)
                
            with open(history_filename + '_val_' + self.mode + '.pkl', 'wb') as f:
                pickle.dump(self.val_history, f)

    def save(self, weights_filename, history_filename = None):
        self.G.save(weights_filename)
      
    def transform(self, in_name, out_name):
        print("Started")
        if self.mode == "RGB":
            data = np.moveaxis(fits.getdata(in_name, ext=0), 0, 2)
        else:
            data = np.moveaxis(np.array([fits.getdata(in_name, ext=0)]), 0, 2)
    
        image = data
        H, W, _ = image.shape
        
        offset = int((self.window_size - self.stride) / 2)
        
        h, w, _ = image.shape
        
        ith = int(h / self.stride) + 1
        itw = int(w / self.stride) + 1
        
        dh = ith * self.stride - h
        dw = itw * self.stride - w
        
        image = np.concatenate((image, image[(h - dh) :, :, :]), axis = 0)
        image = np.concatenate((image, image[:, (w - dw) :, :]), axis = 1)
        
        h, w, _ = image.shape
        image = np.concatenate((image, image[(h - offset) :, :, :]), axis = 0)
        image = np.concatenate((image[: offset, :, :], image), axis = 0)
        image = np.concatenate((image, image[:, (w - offset) :, :]), axis = 1)
        image = np.concatenate((image[:, : offset, :], image), axis = 1)
        
        median = np.median(image[::4,::4,:], axis=[0,1])
        mad = np.median(np.abs(image[::4,::4,:]-median), axis=[0,1])
        
        output = copy.deepcopy(image)
        
        for i in range(ith):
            print(str(i) + " of " + str(ith))
            for j in range(itw):
                x = self.stride * i
                y = self.stride * j
                
                tile = image[x:x+self.window_size, y:y+self.window_size, :]
                #tile_median = np.median(tile, axis=[0,1])
                #tile_mad = np.median(np.abs(tile-tile_median), axis=[0,1])
                tile = (tile - median) / mad * 0.04
                tile_copy = tile.copy()
                tile = np.clip(tile, -1.0, 1.0)
                
                tile = np.expand_dims(tile, axis = 0)
                tile = np.array(self.G(tile)[0])
                
                tile = np.where(tile_copy < 0.95, tile, tile_copy)
                tile = tile / 0.04 * mad + median
                tile = tile[offset:offset+self.stride, offset:offset+self.stride, :]
                output[x+offset:self.stride*(i+1)+offset, y+offset:self.stride*(j+1)+offset, :] = tile
        
        output = np.clip(output, 0, 1)
        output = output[offset:H+offset,offset:W+offset,:]
        
        if self.mode == "RGB":
            self.save_fits(np.moveaxis(output,2,0),out_name,"./")
        else:
            self.save_fits(np.moveaxis(output,2,0)[0],out_name,"./")
            
        print("Finished")
            
    def save_fits(self, image, name, path):
         hdu = fits.PrimaryHDU(image)
         hdul = fits.HDUList([hdu])
         hdul.writeto(path + name + '.fits')       

    def _generator(self):
        return pridnet(self.window_size,self.input_channels)
        #return unet(self.window_size,self.input_channels)
        #return ridnet(self.window_size,self.input_channels)
    
    def downsample(self, filters, size, apply_batchnorm=True):
      initializer = tf.random_normal_initializer(0., 0.02)
    
      result = tf.keras.Sequential()
      result.add(
          tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                 kernel_initializer=initializer, use_bias=False))
    
      if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    
      result.add(tf.keras.layers.LeakyReLU())

      return result
  
    def _discriminator(self):
        
        initializer = tf.random_normal_initializer(0., 0.02)
      
        inp = tf.keras.layers.Input(shape=[self.window_size, self.window_size, 3], name='input_image')
        tar = tf.keras.layers.Input(shape=[self.window_size, self.window_size, 3], name='target_image')
      
        x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)
      
        down1 = self.downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
        down2 = self.downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
        down3 = self.downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)
      
        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                      kernel_initializer=initializer,
                                      use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)
      
        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
      
        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
      
        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)
      
        last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                      kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)
      
        return tf.keras.Model(inputs=[inp, tar], outputs=last)
    


Net = Net(mode = 'RGB', window_size = 256, train_folder = './train', lr = 2e-7, batch_size = 2, stride=128, 
          validation_folder = "./validation/", validation = False)
#Net.load_training_dataset()

Net.load_model('./weights')

#Net.train(99, plot_progress = False, plot_interval = 1000, augmentation=True, save_backups=True, warm_up = False)
#Net.save_model('./weights', './history')

Net.plot_history()
Net.transform("./noisy.fits","denoised")
