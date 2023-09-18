import numpy as np
import torch 
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim


# Defining our custom transformer which converts the image to a tensor and 
# normalizes all data points around 0.5 and 0.5 mean and std. deviation

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])

# Obtaining our training dataset
train_set = datasets.MNIST('./data', download=True, train=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

# Obtaining our testing dataset
eval_set = datasets.MNIST('./data', download=True, train=False, transform=transform)
eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=64, shuffle=True)

# Defining our model

input_size = 784
hidden_sizes = [128, 64]
output_size = 10 # For the 10 digits

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                       nn.ReLU(),
                       nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                       nn.ReLU(),
                       nn.Linear(hidden_sizes[1], output_size),
                       nn.LogSoftmax(dim=1))

print(model)

# Defining our training process

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
epoch_count = 50

criterion = nn.NLLLoss()

for e in range(epoch_count):

    running_loss = 0

    for images, labels in train_loader:

        image_optim = images.view(images.shape[0], -1)
        optimizer.zero_grad()

        output = model(image_optim)
    
        loss = criterion(output, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print("Epoch {} -- Training Loss: {}".format(e, running_loss/len(train_loader)))

correct = 0
all = 0

for images,labels in eval_loader:
  for i in range(len(labels)):

    img = images[i].view(1, 784)
    
    with torch.no_grad():
        logps = model(img)

    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    
    pred_label = probab.index(max(probab))
    true_label = labels.numpy()[i]
    
    if(true_label == pred_label):
      correct += 1
    
    all += 1

print("Number Of Images Tested =", all)
print("\nModel Accuracy =", (correct/all)) 

# Saving the model
torch.save(model, './my_mnist_model.pt') 