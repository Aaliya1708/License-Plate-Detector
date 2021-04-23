import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
from PIL import Image
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1)
        self.pool = nn.MaxPool2d((2,2), stride=(1,1))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(25600, 1024)
        self.fc2 = nn.Linear(1024,128)
        self.fc3 = nn.Linear(128, 36)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 25600)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CharDataset(Dataset):
    def __init__(self, root_dir,transform=None,preprocess=None):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.arr = []
        for class_name in self.classes:
            for ele in os.listdir("/".join([root_dir, class_name])):
                self.arr.append(["/".join([root_dir, class_name,ele]), class_name])
        self.arr = np.array(self.arr)
        np.random.shuffle(self.arr)
        self.transform = transform
        self.preprocess = preprocess
        
    def __len__(self):
        return len(self.arr)
    
    def __getitem__(self, index):
        img = cv2.imread(self.arr[index,0])
        if self.preprocess is not None:
            img = self.preprocess(img)
            
        if self.transform is not None:
            img = self.transform(img)
        
        y_label = torch.tensor((self.classes.index(self.arr[index,1])))
        
        return (img, y_label)

transform = transforms.Compose(
    [
        transforms.Resize((32,32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)
use_transform = transforms.Compose(
    [
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)

def preprocess(img,pad=True,l_size=32, const_pad = False):
    m = max(img.shape[0:2])
    if pad:
        m+=5
    if const_pad:
        img = cv2.copyMakeBorder(img, int((m-img.shape[0])/2),  
                                    int((m-img.shape[0])/2),  
                                    int((m-img.shape[1])/2),  
                                    int((m-img.shape[1])/2),
                                    cv2.BORDER_CONSTANT, None, (255,255,255))
    else:
        img = cv2.copyMakeBorder(img, int((m-img.shape[0])/2),  
                                    int((m-img.shape[0])/2),  
                                    int((m-img.shape[1])/2),  
                                    int((m-img.shape[1])/2),
                                    cv2.BORDER_REPLICATE)
    img = cv2.resize(img, (l_size,l_size))
    return Image.fromarray(img)

def check_accuracy(loader,model):
    accs = 0 
    cnt = 0
    with torch.no_grad():
        for x, y in loader:
            cnt += 1
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            y_pred_softmax = torch.log_softmax(scores, dim = 1)
            _, y_pred = torch.max(y_pred_softmax,dim=-1)
            correct_pred = (y==y_pred).float()
            acc = correct_pred.sum() / len(correct_pred)
            accs += acc
        accs/=cnt
    print("Accuracy:",accs)

def train():
    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    for param_tensor in net.state_dict():
        print(param_tensor, "\t", net.state_dict()[param_tensor].size())
    
    dataset = CharDataset("char_recog_data_1",  transform, preprocess)
    train_set, validation_set = torch.utils.data.random_split(dataset, [16000,3424])

    trainloader = DataLoader(dataset=train_set, shuffle=True, batch_size=32, num_workers=1,pin_memory=True)

    validation_loader = DataLoader(dataset=validation_set, shuffle=True, batch_size=32, num_workers=1,pin_memory=True)
    
    for epoch in range(25):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = net(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i%100 == 99:
                print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
        print("On Training Set")
        check_accuracy(trainloader, net)
        print("On Validation Set")
        check_accuracy(validation_loader,net)
    
    print("Finished training")
    torch.save(net.state_dict(), "character_recog_up.pt")



if __name__=="__main__":
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    train(device)
