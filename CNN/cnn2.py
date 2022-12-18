import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import random_split

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

#Import CIFAR-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

#Get Dataset Loaders
train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=len(trainset), shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=len(testset), shuffle=True)

print(trainset)
print(testset)

batch, lables = iter(train_loader).next()

#Display CIFAR-10
batch.shape
plt.figure(figsize=(15,10))
for i in range(50):
  plt.subplot(5,10,i+1)
  img = np.transpose(batch[i],(1,2,0))
  img.shape
  plt.imshow(img)
plt.savefig('cifar1-.png')

#Get mean and std
print(trainset.data.shape)
data = trainset.data / 255
mean = data.mean(axis = (0,1,2)) 
std = data.std(axis = (0,1,2))
print(f"Mean : {mean}   STD: {std}")

data_t = testset.data / 255
mean_t = data.mean(axis = (0,1,2)) 
std_t = data.std(axis = (0,1,2))
print(f"Mean : {mean_t}   STD: {std_t}")

#Normalize Dataset
train_set_normal = torchvision.datasets.CIFAR10(root='./data', train=True ,download=True
    ,transform=transforms.Compose([
          transforms.ToTensor()
        , transforms.Normalize(mean=mean, std=std)
    ])
)

test_set_normal = torchvision.datasets.CIFAR10(root='./data', train=False ,download=True
    ,transform=transforms.Compose([
          transforms.ToTensor()
        , transforms.Normalize(mean=mean_t, std=std_t)
    ])
)

#Create Validation Set
val_size = 1000
idx = np.arange(len(trainset))

#get indices
trainSize = len(train_set_normal) - val_size
trainIndices= idx[:-val_size]
valIndices = idx[trainSize:]

print(len(valIndices))
print(len(trainIndices))

train_sampler = torch.utils.data.SubsetRandomSampler(trainIndices)
val_sampler = torch.utils.data.SubsetRandomSampler(valIndices)

#Get dataloader with set batch size
batch_size = 32
train_batch_size = batch_size
val_batch_size = batch_size
test_batch_size = batch_size

train_loader = torch.utils.data.DataLoader(train_set_normal, batch_size=train_batch_size, sampler=train_sampler, num_workers=2)
val_loader = torch.utils.data.DataLoader(train_set_normal, batch_size=val_batch_size, sampler=val_sampler, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_set_normal, batch_size=test_batch_size, shuffle=True, num_workers=2)

print(f'Training Set: {len(train_loader) * train_batch_size - 24}')
print(f'Training Set: {len(val_loader) * val_batch_size - 24}')
print(f'Test Set: {len(test_loader) * val_batch_size - 16}')

train_data = next(iter(train_loader))
train_mean = train_data[0].mean()
train_std = train_data[0].std()
print(f'Normalised train Dist: {train_mean}, {train_std}')

test_data = next(iter(test_loader))
test_mean = test_data[0].mean()
test_std = test_data[0].std()
print(f'Normalised test Dist: {test_mean}, {test_std}')

from torch.nn.modules.dropout import Dropout

#Create Model
class ImageClassifier(nn.Module):
  def __init__(self, dropout=False):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 32, 3) #channels\input, filters\output, filter size
    self.conv2 = nn.Conv2d(32, 32, 3)
    self.pool1 = nn.MaxPool2d(2, 2) #filter_size, stride
    self.conv3 = nn.Conv2d(32, 64, 3)
    self.conv4 = nn.Conv2d(64, 64, 3)
    self.pool2 = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(64 * 5 * 5, 512)
    self.fc2 = nn.Linear(512, 10)
    self.softmax = nn.Softmax(1)

    self.dropout = Dropout
    if self.dropout:
      self.dropout1 = nn.Dropout(p=.1) #.1
      self.dropout2 = nn.Dropout(p=.2) #.2
      self.dropout3 = nn.Dropout(p=.5) #.5

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = self.pool1(x)
    if self.dropout: x = self.dropout1(x)
    x = F.relu(self.conv3(x))
    x = F.relu(self.conv4(x))
    x = self.pool2(x)
    if self.dropout: x = self.dropout2(x)
    x = torch.flatten(x,1)
    x = F.relu(self.fc1(x))
    if self.dropout: x = self.dropout3(x)
    x = self.fc2(x)
    #softmax is computed by cross entropy

    return x

  def evaluate(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = self.pool1(x)
    x = F.relu(self.conv3(x))
    x = F.relu(self.conv4(x))
    x = self.pool2(x)
    x = torch.flatten(x,1)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    #softmax is computed by cross entropy

    return x

#Define Model and Loss
learning_rate = 0.001
momentum = 0.9

model = ImageClassifier(dropout=True)
model.to(device)

print(model)
print(f'Torch version: {torch.__version__}')
print(device)
#printed in order of definition NOT order in neural net

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

train_it = iter(train_loader)
images, labels = train_it.next()
print(images.shape)

#Training Loop:
epochs = 20
train_loss = []
train_accuracy = []
val_loss = []
val_accuracy = []
best_val_acc = 0
best_epoch = 0
for epoch in range(1, epochs+1):
  #Declare metrics for each epoch
  running_loss = 0.0
  running_total = 0
  running_correct = 0
  run_step = 0

  #Train Model
  training_loss_last = 0
  training_accuracy_last = 0
  for i, (images, labels) in enumerate(train_loader):
    model.train()

    #Input Shape: (batch_size,1,32,32)
    images = images.to(device)
    labels = labels.to(device) #Shape (B)

    #Forward prop -> backprop -> update
    outputs = model(images) #forward shape (B,10)
    loss = loss_function(outputs, labels)
    optimizer.zero_grad() #reset gradients
    loss.backward() #compute gradients
    optimizer.step() #Update weights

    #Update metrics
    running_loss += loss.item()
    running_total += labels.size(0)

    #Evaluate predictions
    with torch.no_grad():
      _, predicted = outputs.max(1)
    running_correct += (predicted == labels).sum().item()
    run_step += 1

    #Print metrics after every 200 iterations
    #Update 
    if i % 200 == 0:
      print(f'epoch {epoch}, steps: {i}, '
            f'train_loss: {running_loss / run_step :.3f} '
            f'running_accuracy: {100 * running_correct / running_total :.1f}%')
      training_loss_last = running_loss / run_step
      training_accuracy_last = 100 * running_correct / running_total

      #reset metrics
      running_loss = 0.0
      running_total = 0
      running_correct = 0
      run_step = 0
  
  #Append last training loss and accuracy values after each training epoch 
  train_loss.append(training_loss_last)
  train_accuracy.append(training_accuracy_last)

  #Validate Model
  correct = 0
  total = 0
  v_loss = 0
  val_acc = 0
  model.eval()
  with torch.no_grad():
    for data in val_loader:
      #Get data
      images, labels = data
      images, labels = images.to(device), labels.to(device)

      #Evaliate model
      outputs = model(images)
      _, predicted = outputs.max(1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item() 
      v_loss = running_loss / run_step
      val_acc = 100 * correct / total

      #Update metrics
      running_loss += loss.item()
      run_step += 1

  #Append last validation loss and accuracy values after each training epoch 
  val_loss.append(v_loss)
  val_accuracy.append(val_acc)

  #Set optimal values
  if best_val_acc < val_acc:
    best_val_acc = val_acc
    best_epoch = epoch
  
  print(f'\nEpoch: {epoch} '
        f'Validation loss: {running_loss / run_step :.3f} '
        f'Validation accuracy: {100 * correct / total :.2f}% ')

print('\nTraining Complete\n')
print(f'Best epoch: {best_epoch} '
      f'with highest validation accuracy: {best_val_acc}% \n')

plt.figure(figsize=(10,5))

#80.4%

#Plot train loss/accuracy
plt.subplot(1,2,1)
plt.plot(range(len(train_loss)), train_loss, color='blue', label='Training')
plt.plot(range(len(val_loss)), val_loss, color='red', label='Validation')
plt.legend()
plt.title(f'Loss on the CNN')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.subplot(1,2,2)
plt.plot(range(len(train_accuracy)), train_accuracy, color='blue', label='Training')
plt.plot(range(len(val_accuracy)), val_accuracy, color='red', label='Validation')
plt.legend()
plt.title(f'Accuracy on the CNN')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.savefig('loss.png')

#71.23
 #73.11, 71.84, 56.09
 #78.78

#Test Set Validation
correct = 0
total = 0
model.eval()
with torch.no_grad():
  for data in test_loader:
    #Get data
    images, labels = data
    images, labels = images.to(device), labels.to(device)

    #Evaluate model
    outputs = model.evaluate(images)
    _, predicted = outputs.max(1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
  
print(f'Test accuracy: {100 * correct / total}%')


#Print classified images
plt.figure(figsize=(10,5))

n = 20
num_ims = 0
plt.figure(figsize=(15,10))

batch = []
preds = []
model.eval()
with torch.no_grad():
  batch, labels = iter(val_loader).next()
  inputs = batch.to(device)
  labels = labels.to(device)

  outputs = model(inputs)
  _, preds = torch.max(outputs, 1)

batch = transforms.Normalize(mean=-mean/std, std=1/std)(batch)
for i in range(n):
  plt.subplot(5,10,i+1)
  img = np.transpose(batch[i],(1,2,0))
  img.shape
  plt.imshow(img)
  plt.title(f'{classes[preds[i]]}')
plt.savefig('test.png')