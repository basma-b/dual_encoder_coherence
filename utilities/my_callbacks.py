from __future__ import division
import keras
import numpy as np
from utilities.data_helper import compute_recall_ks

class Histories(keras.callbacks.Callback):
    def on_train_begin(self, logs={}): 
        self.accs = []
        self.losses = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        y_pred = self.model.predict(([self.validation_data[0], self.validation_data[1], self.validation_data[2]]))
        print (y_pred)
        recall_k = compute_recall_ks(y_pred[:,0])
        
        self.accs.append(recall_k[10][1]) # not sure if i'll keep it

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
    