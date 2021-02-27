import keras
import numpy as np
import time
import pandas as pd
import cmath
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
from sklearn import decomposition
from keras.optimizers import SGD


class cyclops:
    def __init__(self, input_width):
        
        """
        Input Layer
        """
        input_layer = keras.layers.Input(shape=(input_width,), name='input_layer')
        

        """
        kernel_initializer is the statistical distribution used to initialise the network weights. 
        (Glorot/random, uniform/normal)
        """
        kernel_initializer = keras.initializers.glorot_normal(seed=None)
        
        
        """
        a0 and a1 are densely-connected layers, with weights matrices (kernels) initialised according to the
        chosen kernel initializer, and initial biases of zero.
        """
        a0 = keras.layers.Dense(name='encoder_circular_in_0',
                                 units=1,
                                 kernel_initializer=kernel_initializer,
                                 bias_initializer='zeros'
                                 )(input_layer)
        a1 = keras.layers.Dense(name='encoder_circular_in_1',
                                 units=1,
                                 kernel_initializer=kernel_initializer,
                                 bias_initializer='zeros'
                                 )(input_layer)
        
        
        """
        The Lambda function allows us to carry out operations on the tensors. 
        In this case, keras.backend.square squares every element in the tensor.
        """
        a02 = keras.layers.Lambda(keras.backend.square, name='a0_sqr')(a0)
        a12 = keras.layers.Lambda(keras.backend.square, name='a1_sqr')(a1)
        
        
        """
        Performs sum of the tensors a02 and a12
        """
        aa = keras.layers.Add(name='sqr_len')([a02, a12])
        
        sqrt_aa = keras.layers.Lambda(keras.backend.sqrt, name='len')(aa)
        
        
        """
        Normalisation to complete the circular constraint steps
        """
        a0_ = keras.layers.Lambda(lambda x: x[0] / x[1], name='encoder_circular_out_0')([a0, sqrt_aa])
        a1_ = keras.layers.Lambda(lambda x: x[0] / x[1], name='encoder_circular_out_1')([a1, sqrt_aa])

        
        x = keras.layers.Concatenate(name='embedding')([a0_, a1_])
        
        
        """
        Output layer
        """
        y = keras.layers.Dense(name='output_layer',
                                   units=input_width,
                                   kernel_initializer=kernel_initializer,
                                   bias_initializer='zeros'
                                   )(x)

        self.model = keras.Model(outputs=y, inputs=input_layer)
        
        
    class MyCallback(keras.callbacks.Callback):
        """
        Function to report loss and time taken per series of epochs
        """
        def __init__(self, interval):
            super().__init__()
            self.cnt = 0
            self.interval = interval
            self.start_time = 0
            self.rec = {'time': [], 'loss': []}

        def on_train_begin(self, logs=None):
            self.start_time = time.time()

        def on_epoch_end(self, batch, logs=None):
            self.cnt += 1
            self.rec['time'].append(time.time() - self.start_time)
            self.rec['loss'].append(logs.get('loss'))
            if self.cnt % self.interval == 0:
                print(f'epoch: {self.cnt}/{self.params["epochs"]}, loss: {logs.get("loss") : .4f}, '
                      f'time elapsed: {self.rec["time"][-1] : .2f}s, '
                      f'time left: {((self.params["epochs"] / self.cnt - 1) * self.rec["time"][-1]) : .2f}s') 
                

    def train(self, data, batch_size=10, epochs=100, verbose=10, rate=0.3, momentum=0.5):
        """
        Train the model. It will not reset the weights each time so it can be called iteratively.
        data: data used for training
        batch_size: batch size for training, if unspecified default to 32 as is set by keras
        epochs: number of epochs in training
        verbose: per how many epochs does it report the loss, time consumption, etc.
        rate: training rate
        """

        opt = SGD(lr=rate, momentum=momentum)

        self.model.compile(loss='mean_squared_error',
                           optimizer=opt)
        
        my_callback = self.MyCallback(verbose)
        history = self.model.fit(data, data, batch_size=batch_size, epochs=epochs, verbose=0, callbacks=[my_callback])

        return history
    
    
    def predict_pseudotime(self, data):
        res = keras.backend.function(inputs=[self.model.get_layer('input_layer').input],
                                     outputs=[self.model.get_layer('embedding').output]
                                     )([data])
        return np.arctan2(res[0][:, 0], res[0][:, 1])

    
    def z_p(self, data):
        res = keras.backend.function(inputs=[self.model.get_layer('input_layer').input],
                                     outputs=[self.model.get_layer('embedding').output]
                                     )([data])
        return res[0][:, 0]
    
    
    def z_q(self, data):
        res = keras.backend.function(inputs=[self.model.get_layer('input_layer').input],
                                     outputs=[self.model.get_layer('embedding').output]
                                     )([data])
        return res[0][:, 1]
    
 
    def phase_output(self, data):
        """
        Outputs a clean dataframe with phase values for each time sample
        """
        d = {'z_p':self.z_p(data),
             'z_q':self.z_q(data),
             'pseudotime':self.predict_pseudotime(data)}
             
        df_phase = pd.DataFrame(d)
       
        df_phase = df_phase.sort_values(by='pseudotime', ascending=True)
        
        return df_phase
    
    
    @staticmethod
    def plot_polar(nc, phase_idx, phase_list):
        """
        Outputs a polar plot of phase values
        """
        plt.figure(figsize=(10,8))
        for i in range(phase_idx.shape[0]):
            if i <= nc/2:
                plt.polar([0,phase_list[i]],[0,1],marker='o', label=phase_idx[i])
            else:
                plt.polar([0,phase_list[i]],[0,1],marker='x', label=phase_idx[i])
                
        plt.legend(loc=[1.1,0.1])
        plt.show()
        
        
    @staticmethod
    def plot_phase_time(phase_idx, phase_list):
        """
        Outputs a plot of CYCLOPS phase against sample collection times
        """
        plt.figure(figsize=(10,8))
        plt.scatter(phase_idx, phase_list)
        plt.title('Plot of CYCLOPS phase vs sample collection times', size=20)
        plt.xlabel('Sample collection (CT)', size=15)
        plt.ylabel('CYCLOPS Phase (rad)', size=15)
        plt.show()
        
        
    @staticmethod
    def remap(nc, phase_idx):
        remap_dict = {}
        for i in range(nc):
            remap_dict.update({phase_idx[i]+1:i+1})
        return remap_dict
