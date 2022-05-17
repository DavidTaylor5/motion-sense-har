
from logging import raiseExceptions
from msilib.schema import Binary
import os
import math

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import flwr as fl
import tensorflow as tf

import harCNN
import dataPreprocess

import numpy as np
#import commonfinal as common

# I need to make sure that my federated learning settings are reproducible
# I need to create a 'seed' for both python random and tf random
#####################################################################
RANDOM_SEED = 47568
#seed(47568
# #tf.random.set_random_seed(seed_value))
os.environ['PYTHONHASHSEED']=str(47568)
#random.seed(47568)
tf.random.set_seed(47568)

np.random.seed(RANDOM_SEED)

#from tensorflow.python.keras.layers import normalization
#cannot import name 'normalization' from 'tensorflow.python.keras.layers' (C:\Users\David\anaconda3\envs\keras-gpu\lib\site-packages\tensorflow\python\keras\layers\__init__.py)
#tensorflow 2.6 works 2.8 doesn't?
# I need to add the privacy imports (when do I use Vectorized (parallel?) optimizer over regular optimizer)
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras_vectorized import VectorizedDPKerasSGDOptimizer
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer

from tensorflow.python.keras.losses import BinaryCrossentropy
from tensorflow.python.keras.losses import CategoricalCrossentropy #I'm working with categorical crossentropy
#add a PRIVACY_LOSS value
PRIVACY_LOSS = 0

import common
#####################################################################


NUM_CLIENTS = 24

run_args = {
    'batch_size':24,
    'local_epochs':10,
    'dpsgd':False,
    'microbatches':24, #what is a microbatch
    'noise_multiplier':1.1, #what is a noise multiplier
    '12_norm_clip':1.0,
    'learning_rate':.1

}

print(run_args.get('batch_size'))

class FlwerClient(fl.client.NumPyClient):
    def __init__(self, model, cid, x_train, y_train, x_val, y_val) -> None:
        super().__init__()
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.cid = cid
        self.acc = []
        self.loss = []
        #model also needs to know batch_size, local_epochs, and dpsgd
        #############################
        self.batch_size = run_args.get('batch_size')
        self.local_epochs = run_args.get('local_epochs')
        self.dpsgd = run_args.get('dpsgd')
        ############################# #how many clients should I train with per round, how many should I evaluate at end of round?

        ################################
        if run_args.get('dpsgd'):
            self.noise_multiplier = run_args.get('noise_multiplier')
            if run_args.get('batch_size') % run_args.get('microbatches') != 0:
                raise ValueError("Number of microbatches should divide evenly batch_size")
            optimizer = VectorizedDPKerasSGDOptimizer(
                l2_norm_clip=run_args.get("l2_norm_clip"),
                noise_multiplier=run_args.get("noise_multiplier"),
                num_microbatches=run_args.get("microbatches"),
                learning_rate=run_args.get("learning_rate")
            )
            #opt DPKerasSGD
            #loss = BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE) #change to categorical crossentropy
            loss = CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        else:
            optimizer = tf.keras.optimizers.SGD(learning_rate=run_args.get("learning_rate")) #change to categorical crossentropy
            #loss = BinaryCrossentropy(from_logits=True)
            loss = CategoricalCrossentropy(from_logits=True)
        
        #compile model with Keras #"binary_crossentropy"
        model.compile(optimizer=optimizer, loss=loss, metrics=[tf.keras.metrics.Recall()])


        ################################
    def get_parameters(self):
        return self.model.get_weights() #why do I take this out?
        #raise Exception("Not implemented (server-side parameter initialization)") #why?
        #ray::launch_and_get_parameters()
        #Exception: Not implemented (server-side parameter initialization)

#C:\Users\David\AppData\Roaming\Python\Python39\site-packages\tensorflow\python\keras\backend.py:5029: 
# UserWarning: "`binary_crossentropy` received `from_logits=True`, but the `output` argument was produced by a sigmoid or softmax activation and 
# thus does not represent logits. Was this intended?"
#(launch_and_fit pid=24660) 39/39 - 0s - loss: 0.4361 - recall_1: 0.0000e+00 - 128ms/epoch - 3ms/step #why recall instead of accuracy
#recall metric is the recall -> disparate impact (TP), 

#grab the server from the model at the end train over all participants test data
#confusion matrix based on sensitive attributes
#confusion matrix based on sensitive attributes

#1) train and test split
#2) train in central paradigm
"""
#unified test set available
1) 

"""

    def fit(self, parameters, config):

        #################################
        """Train parameters on the locally held training set."""
        #update local model parameters
        global PRIVACY_LOSS
        if self.dpsgd:
            privacy_spent = common.compute_epsilon( #can't I compute this outside of the fl?
                self.local_epochs,
                len(self.x_train), 
                self.batch_size,
                self.noise_multiplier,
            )
            PRIVACY_LOSS += privacy_spent
            print("privacy_spent -> ", privacy_spent)
        #################################

        self.model.set_weights(parameters)

        self.model.fit(self.x_train, self.y_train, epochs=self.local_epochs, batch_size=self.batch_size, verbose=2) #no need to specify validation data

        return self.model.get_weights(), len(self.x_train), {}
    
    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)

        loss, acc = self.model.evaluate(self.x_val, self.y_val, verbose=2)

        print("client " + str(self.cid) + " -> is working!")
        
        try: 
            f = open("FLDP/participant" + self.cid + ".txt", "a")
            f.write("write something!")
            f.write("My cid -> " + self.cid + " Loss: " + str(loss) + " Accuracy: " + str(acc) + "\n")
            f.close()
        except:
            print("STILL MESSING UP THE OPEN FILE STUFF!")

        try:
            self.acc.append(acc)
            print("At least I got to print the accuracy tho! -> " + str(self.acc[-1]))
        except:
            print("ERROR ON ACCESSING THE ACC LIST?")

        return loss, len(self.x_val), {"accuracy":acc}



# necessary data below, window data for each participant
partData = dataPreprocess.getIndividualDatasets(24)
dataPreprocess.normalizeParticipants(partData)
pooledData = dataPreprocess.getCentralDataset(partData)

part_windows = harCNN.participantWindows(partData, 50)

print(len(part_windows))



def client_fn(cid: str) -> fl.client.Client:
    # Load model
    model = harCNN.sensor_activity(n_timesteps=50, n_features=12, n_outputs=6) #as specified by David M

    (x_train, y_train, x_val, y_val) = part_windows[int(cid)]

    # Create and return client
    return FlwerClient(model, cid, x_train, y_train, x_val, y_val)

# experiemental evaluate_config for clients
def evaluate_config(rnd: int): #EXPERIMENTAL
    val_steps = 5 if rnd < 4 else 10
    return {"val_steps":val_steps}

def main() -> None:

    #get participant data in windows

    # Start Flower simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        client_resources={"num_cpus":2},
        num_rounds=20,
        strategy=fl.server.strategy.FedAvg(
            fraction_fit=0.1,
            min_fit_clients=10, #testing, only fitting 4 participants per round #fitting with 10 clients a round
            min_available_clients=NUM_CLIENTS,
            fraction_eval=0.2,
            min_eval_clients=24,
            on_evaluate_config_fn=evaluate_config
        )
    )

if __name__ == "__main__":

    # f = open("FlwrCNNResult/participant" + str(1) + ".txt", "a")
    # f.write("write something!")
    # f.write("My cid -> " + str(1) + " Loss: " + str(.5) + " Accuracy: " + str(.3) + "\n")
    # f.close()
    # print("done with evaluation!")


    main()

    if run_args.get("dpsgd"):
        print("Privacy Loss: ", PRIVACY_LOSS)

    #is privacy_loss global to all clients? how does this work? do they all have access to the same PRIVACY_LOSS variable?
    
