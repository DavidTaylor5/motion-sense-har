import os
import math

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import flwr as fl
import tensorflow as tf

import harCNN
import dataPreprocess

NUM_CLIENTS = 24

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


    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)

        self.model.fit(self.x_train, self.y_train, epochs=20, verbose=2)

        return self.model.get_weights(), len(self.x_train), {}
    
    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)

        loss, acc = self.model.evaluate(self.x_val, self.y_val, verbose=2)

        print("client " + str(self.cid) + " -> is working!")
        
        try: 
            f = open("fedResults/participant" + self.cid + ".txt", "a")
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
    
