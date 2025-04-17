import argparse
import warnings
from collections import OrderedDict
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import prepare_dataset
import random
import logging

# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

# Configurer les logs
logging.basicConfig(level=logging.INFO, format='%(message)s')

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    """Model"""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train(net, trainloader, epochs, attack_type):
    """Train the model on the training set."""

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    NUM_CLASSES = 10 # CIFAR10 has 10 classes
    for _ in range(epochs):
        for images, labels in tqdm(trainloader, "Training"):


            # Appliquer l'attaque d'inversion d'étiquettes avec 50 % de probabilité
            if attack_type == "label_flipping" and random.random() < 0.5: # 50 % de probabilité que l'attaquant exécute cette attaque et 50 % de probabilité qu'il suive le protocole FedAvg
                labels = (labels + 1) % NUM_CLASSES  # Décalage circulaire des étiquettes

            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()

            if attack_type == "model_poisoning" and random.random() < 0.5: # 50 % de probabilité que l'attaquant exécute cette attaque et 50 % de probabilité qu'il suive le protocole FedAvg
                # Inversion des gradients (montée de gradient)
                for param in net.parameters():
                    if param.grad is not None:
                        param.grad *= -1
                        param.grad = torch.clamp(param.grad, -1.0, 1.0)

            
            optimizer.step()



def test(net, valloader):
    """Validate the model on the validation set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(valloader, "Testing"):
            outputs = net(images.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(valloader.dataset)
    return loss, accuracy




# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Get node id
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--node_id",
    choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    required=True,
    type=int,
    help="Client id",
)
parser.add_argument(
    "--n",
    type=int,
    default=5,
    help="The number of clients in total",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="CIFAR10",
    help="Dataset: [CIFAR10]",
)
parser.add_argument(
    "--data_split",
    type=str,
    default="iid",
    help="iid",
)
parser.add_argument(
    "--local_epochs",
    type=int,
    default=1,
    help="époques",
)
parser.add_argument(
    "--attack_type",
    type=str,
    default="label_flipping",
    help="Types des attaques: [label_flipping, model_poisoning]"
)
cid = parser.parse_args().node_id
n = parser.parse_args().n
data_split = parser.parse_args().data_split
dataset = parser.parse_args().dataset
local_epochs = parser.parse_args().local_epochs
attack_type = parser.parse_args().attack_type

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid):
        self.cid = cid
        self.net = Net().to(DEVICE)
        self.trainloader, self.valloader, _ = prepare_dataset.get_data_loader(n, cid, data_split=data_split, dataset=dataset)
        self.history = {'accuracy': []}  # Historique pour stocker l'accuracy

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # Read values from config
        server_round = config["server_round"]
        logging.info(f"Round {server_round} - Client {self.cid} training started...")
        # Use values provided by the config
        print(f"[Client {self.cid}, round {server_round}] fit")
        self.set_parameters(parameters)
        train(self.net, self.trainloader, epochs=local_epochs, attack_type=attack_type)
        # Calculer la précision après l'entraînement
        _, accuracy = test(self.net, self.valloader)
        # Enregistrer l'accuracy dans l'historique
        self.history['accuracy'].append((server_round, accuracy))
        # Affichage du round et de l'accuracy
        logging.info(f"Round {server_round}: Accuracy: {accuracy:.4f}")
        # Affichage de l'historique des précisions pour le client
        logging.info(f"Client {self.cid} History: {self.history['accuracy']}")
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def get_history(self):
        return self.history

# Start Flower client
fl.client.start_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(cid).to_client(),
)