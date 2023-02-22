import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
import datetime
import sys
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

## recall moving tensors to cpu gives a better description of errors in some cases

## for saveing model parameters after training
save_path = r'C:\Users\M\OneDrive - Carleton University\Documents\my_stuff\Projects\Kaggle_Digit_Recognizer\DigitClassifier_saved_model.pt'

## setting device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('Device in use: {} \n'.format(device))

torch.manual_seed(666) 
learning_rate = 0.001
n_epochs = 10
batch_size = 32     ## takes about 25mins for one epoch with batch size 32



## first part is loading & handling then data
image_train_flatenned = pd.read_csv(r'C:\Users\M\OneDrive - Carleton University\Documents\my_stuff\Projects\Kaggle_Digit_Recognizer\train.csv')
labels = image_train_flatenned.label
labels = labels.to_numpy()
image_train_flatenned = image_train_flatenned.drop(['label'], axis=1)


## func for viewing images, to test how data looks , img_num is the row number in the matrix of greyscale images
def view_img(img_num):
    num_array = np.array(image_train_flatenned.iloc[img_num])
    num_array = np.reshape(num_array, (28, 28))
    
    ## need to do this next step for this PIL.Image library to work
    num_array = (num_array * 255).astype(np.uint8)
    image = Image.fromarray(num_array, mode = 'L')
    image.show()



## converting numpy arrays to tensors, and turning on their derivatives in computational graph if needed, (not putting on gpu for now, so we don't load whole training dataset on it, will move them later when inside the training loop)
images_tensor = image_train_flatenned.to_numpy()
images_tensor = np.reshape(images_tensor, (len(image_train_flatenned), 28, 28))
images_tensor =  torch.from_numpy(images_tensor).float()
images_tensor = images_tensor.view(len(image_train_flatenned),1,28,28)               ## very important to do this to math NCHW format (batch size, no. of channels, height, width)
images_tensor.requires_grad = True

labels = torch.from_numpy(labels)


############################################
## Due to relative simplicity of what we're doing here, we don't really need to make a custom class, can use built-in TensorDataset class
# class CustomDataset(Dataset):
#     def __init__(self, images, labels):
#         self.x = images
#         self.y = labels
        
#     def __getitem__(self, index):
#         return (self.x[index], self.y[index])

#     def __len__(self):
#         return len(self.x)
############################################


full_train_dataset = TensorDataset(images_tensor, labels)

## splitting into train and validation datasets
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size,val_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size*2)   



## defining the model class
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__() 
        
        ## this function and its calls below to calculate dim for linear layer (the order of calculations needs to match the order in the forward method defined below), idk if there's cleaner built-in pytorch way. Using [(image-filter+2*padding)/stride]+1 to get ouput dim of conv layers, and [(image-filter)/stride]+1 to get ouput dim of maxpool layers, then repeated application    
        def calculate_dim(self, I, F, S, P, convORpool):
            if convORpool == 'conv':
                return ((I - F + 2*P)/S)+1
            if convORpool == 'pool':
                return ((I - F)/S)+1

        self.i1 = 28    
        self.f1 = 5     
        self.s = 1
        self.p = 0                      ## we can use zero-padding to keep the area of the image fixed after each convolution, if we want a "very" deep CNN, so the image doesn't shrik too much when it reaches later layers
        self.pool = 2
        
        ## after 1st conv layer:
        self.i2 = calculate_dim(self, I=self.i1, F=self.f1 , S=self.s, P=self.p, convORpool='conv')
        
        ## after 1st maxpool: (default stride for max pool is the same as kernel size)
        self.i3 = calculate_dim(self, I=self.i2, F=self.pool , S=self.pool, P=self.p, convORpool='pool')
        self.f2 = 3
        
        ## after 2nd conv layer:
        self.i4 = calculate_dim(self, I=self.i3, F=self.f2 , S=self.s, P=self.p, convORpool='conv')
        
        ## after 2nd maxpool (input to linear layer):
        self.linear_dim = calculate_dim(self, I=self.i4, F=self.pool , S=self.pool, P=self.p, convORpool='pool')
        self.linear_dim = int(self.linear_dim)          ## needs to be int to pass as size into neural network layers
        

   
   ## first number is input, our images are black&white so only 1 color channel, middle number is number of output features from convolution, 5x5 is the filter
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=self.f1)                             ## convolution leverages sparsness (because kernel is smaller than input image), parameter sharing (because same learned kernel element is used for more than one pixel) which acts as a form of regularization (leads to better generalization), and equivariant representations (at least with respect to shifts), so that say if an edge is shifteed in the input, the ouput changes the same way each time, corresponding to that shift
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=20, kernel_size=self.f2)
        self.pool = nn.MaxPool2d(self.pool)                                                                          ## helps the representation be invariant to tmall translations, in cases where we care about exactly where the features are, shouldn't include this
        self.lin1 = torch.nn.Linear(in_features=20*self.linear_dim*self.linear_dim, out_features=120, bias=True) 
        self.lin2 = torch.nn.Linear(in_features=120, out_features=84)
        self.lin3 = torch.nn.Linear(in_features=84, out_features=10)
       ## add droput layers later if performance isn't satisfacotry, also acts a form of regularization (which can be good, if model is sufficiently large)
   
       
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))         ## conv1 outputs (6, 24x24 images), maxpool outputs (6, 23x23 images)
        x = self.pool(F.relu(self.conv2(x)))         ## conv2 outputs (20, 21x21 images), maxpool outputs (20, 20x20 images)
        x = x.view(-1, self.num_flat_features(x))                 ## flattens the tensor, to input into linear layer
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x
    
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
            
        # print ('num_features: {} \n'.format(num_features))
        return num_features

      
    
def make_train_step(model, loss_func, optimizer):
    
    def train_step(x, y):
        model.train()           ## put model in train mode
        y_hat = model(x)
        
        loss = loss_func(input=y_hat, target=y)
        loss.backward()         ## calculate gradients
        optimizer.step()        ## update parameters
        optimizer.zero_grad()   ## zero gradients  (because they accumulate)

        return loss.item()
    return train_step   



model = DigitClassifier()  
model.to(device)  
loss_func = nn.CrossEntropyLoss()         ## applies softmax to the outputs to scale them from 0 to 1 (and they sum to 1 so they represent probabilities) before calcualting the loss
optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate) 

train_step = make_train_step(model, loss_func, optimizer)


## defning the training loop
losses = []
val_losses = []
val_accuracy = []


def start_train_loop(save_model=True):
    print('start time is: {} \n'.format(datetime.datetime.now()))
    for epoch in range(n_epochs):
        
        count = 0
        
        for x_batch, y_batch in train_loader:
            count += 1
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            loss = train_step(x_batch, y_batch)
            losses.append(loss)
            
           
            
            if count % 50 == 0:
                with torch.no_grad():                       ## turn off gradient computation
                
                    correct = 0
                    total = 0
                    val_loss = 0
                    
                    for x_val, y_val in val_loader:
                        x_val = x_val.to(device)
                        y_val = y_val.to(device)
                        
                        model.eval()                        ## put model in evaluation mode
            
                        y_hat = model(x_val)
                        val_loss += loss_func(input=y_hat, target=y_val)
                        
                        
                        predicted = torch.max(y_hat.data, 1)[1]
                        total += len(y_val)
                        correct += (predicted == y_val).sum()
                        
                    accuracy = 100 * correct / float(total)
                    val_loss = val_loss / float(total)
                        
                    val_accuracy.append(accuracy)
                    val_losses.append(val_loss.item())
                    
                    print ('[epoch: {}] train_loss: {}, val_accuracy: {}, avg_val_loss: {} \n'.format(epoch+1, loss, accuracy, val_loss))
    
    if save_model == True:
        torch.save(model.state_dict(), save_path)
    print('Finsih time is: {} \n'.format(datetime.datetime.now()))


## call training loop
# start_train_loop()

# print(model.state_dict())



def plot_losses():
    plt.plot(losses, label = 'train_loss')
    plt.legend(loc='upper left', frameon=True, edgecolor='k')
    plt.show()
    

    plt.plot(val_losses, label = 'validation_loss')
    plt.legend(loc='upper left', frameon=True, edgecolor='k')
    plt.show()
    
    
    ## fix this part: (TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.)

    try:
        val_accuracy.cpu()
    except:
        print('error1')
    
    try:
        val_accuracy = val_accuracy.numpy()
    except:
        print('error2')
        
    plt.plot(val_accuracy, label = 'validation accuracy')
    plt.legend(loc='upper left', frameon=True, edgecolor='k')
    plt.show()




