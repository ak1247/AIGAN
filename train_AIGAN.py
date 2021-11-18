import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
## My functions
from models import MNIST_target_net
from AIGAN import AIGAN

## Hyperparameters
BATCH_SIZE = 128
EPOCHS = 300


device = torch.device("cuda:3")
print("Working device:- {}".format(device))
print("Loading pretrained model...")
pretrained_model = "../../MNIST_target_model.pth"
target_model = MNIST_target_net().to(device)
target_model.load_state_dict(torch.load(pretrained_model))
print("Loaded.")
target_model.eval()

### Data
## Train Dataset
dataset = datasets.MNIST('../../dataset',train=True,transform=transforms.ToTensor(),download=True)
DATASET_CHANNELS = 1
dataloader = DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=1)


aiGAN = AIGAN(device,target_model,DATASET_CHANNELS)
aiGAN.train(dataloader,EPOCHS)
