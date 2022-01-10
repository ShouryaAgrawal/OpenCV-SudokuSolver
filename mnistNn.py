
import torch.utils.data
from torch.utils.data.dataset import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import cv2
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import copy


#### Different Dataset

path = 'data/myData/myData' #custom data

myL = os.listdir(path)
# print(myL)
numCl = len(myL)

all_img = []
all_img_class = []
for i in range(len(myL)):
    inFolder = os.listdir(path+"/" + str(i) )
    for j in inFolder:
        tempImg = cv2.imread(path+"/" + str(i)+ "/"+ j)
        gr = cv2.cvtColor(tempImg,cv2.COLOR_BGR2GRAY)
        reImg = cv2.resize(gr,(28,28))
        all_img.append(reImg)
        all_img_class.append(i)
    # print(str(i),end = " ")
# print(len(all_img))
# print(len(all_img_class))   DEBUGGING

all_img = np.array(all_img)
all_img_class = np.array(all_img_class)

# print(all_img.shape)
# print(all_img_class.shape) DEBUGGING
# print('done')


class MyCustomDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
    # stuff

    def __getitem__(self, index):
        # stuff
        retimg = all_img[index]
        if self.transform is not None:
            retimg = self.transform(all_img[index])

        return (retimg,all_img_class[index])

    def __len__(self):
        return len(all_img)

## Preprocessing our data

custom_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # These values are fot the mnist dataset but work here as well
])

ds = MyCustomDataset(custom_transform)
#print(ds.__getitem__(4))  for debugging

total_count = len(ds)
train_count = int(0.7 * total_count)
valid_count = int(0.1* total_count)
test_count = total_count - train_count - valid_count
train_set, valid_set, test_set = torch.utils.data.random_split(ds, (train_count, valid_count,test_count))

tr_loader = torch.utils.data.DataLoader(train_set, batch_size=100,shuffle=True)
tv_loader = torch.utils.data.DataLoader(valid_set, batch_size=100,shuffle=True)
ts_loader = torch.utils.data.DataLoader(test_set, batch_size=100,shuffle=True)


#
# #CNN Time
# ## LeNet
def create_model():
    model = nn.Sequential(
        nn.Conv2d(in_channels = 1,out_channels = 6, kernel_size = 5, padding = 2),
        nn.ReLU(),
        nn.MaxPool2d( 2 ,stride= 2),
        nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0),
        nn.ReLU(),
        nn.MaxPool2d( 2, stride=2),
        nn.Flatten(),
        nn.Linear(400, 120),
        nn.ReLU(),
        nn.Linear(120,84),
        nn.ReLU(),
        nn.Linear(84, 10)
        )
    return model
def validate(model, data):
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(data):
        x = model(images)
        value, pred = torch.max(x,1)
        pred = pred.data.cpu()
        total += x.size(0)
        correct += torch.sum(pred == labels)
    return correct*100./total



def train_model(n_epoch = 3, lr = 1e-3, device = "cpu"  ):

    cnn= create_model().to(device)
    cec = nn.CrossEntropyLoss()
    opt = optim.Adam(cnn.parameters(), lr= lr)
    max_accuracy = 0
    for epoch in range(n_epoch):
        acc = []
        losses = list()
        for i, (images,labels) in enumerate(tr_loader):
            images = images.to(device)
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)

            opt.zero_grad()
            pred = cnn(images)
            loss = cec(pred,labels)
            loss.backward()
            opt.step()
            losses.append(loss)

        #print(f'Epoch {epoch+1}, train loss: {torch.tensor(losses).mean():.2f}')  # Debugging and validating
        accuracy = float(validate(cnn, tv_loader))
        acc.append(accuracy)
        if accuracy > max_accuracy:
            best_model = copy.deepcopy(cnn)
            max_accuracy = accuracy
            # print("Saving Best Model with Accuracy: ", accuracy)
            # print('Epoch:', epoch + 1, "Accuracy :", accuracy, '%')
        # plt.plot(acc)
    return  best_model


def pred_model(model,data):
    y_pred = []
    y_true = []
    for i, (images, labels) in enumerate(data):
        x = model(images)
        value, pred = torch.max(x, 1)
        pred = pred.data.cpu()
        y_pred.extend(list(pred.numpy()))
        y_true.extend(list(labels.numpy()))
    return np.array(y_pred), np.array(y_true)


def run_model(img,model, device = "cpu"):
    with torch.no_grad():

        imgr = cv2.resize(img, (28, 28))
        pr = model(torch.unsqueeze(custom_transform(imgr), axis=0).float().to(device))
        return F.softmax(pr, dim = -1).cpu().numpy()
#
# md = train_model(15)
# torch.save(md.state_dict(), "md.pth")


md = create_model().to("cpu")
md.load_state_dict(torch.load("md.pth"))

#
y_pred, y_true = pred_model(md, ts_loader)

conf = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=np.arange(0,10)))

#print(conf) If you need to check



