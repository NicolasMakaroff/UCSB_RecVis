import zipfile
import os

import torchvision.transforms as transforms

# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 64 x 64 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set
data_transforms = transforms.Compose([
    transforms.Resize((341, 341)),
    transforms.ToTensor(),                    
    transforms.RandomHorizontalFlip(),  # horizontaly flip the images with probability 0.5
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(45),
    transforms.RandomCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

#for the test set 
data_transforms_test = transforms.Compose([
    transforms.Resize((341,341)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])



