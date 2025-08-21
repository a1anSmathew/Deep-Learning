# Dataset stores the samples and their corresponding labels,
# and DataLoader wraps an iterable around the Dataset to enable easy access to the samples.


import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
import pandas as pd
from torchvision.io import decode_image
from torch.utils.data import DataLoader


training_data = datasets.FashionMNIST( #Training_data will hold the training dataset and FashionMNIST is a preset dataset
    root="data", #This tells the loader to store or look for the dataset in the folder named "data". If the folder doesn't exist, it will be created automatically.
    train=True,
    download=True,
    transform=ToTensor() #Converts images into Tensors
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

labels_map = { #Creates a dictionary which maps numeric values to human readable names
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8)) #Creating a blank canvas
cols, rows = 3, 3 #Grid of total 9 images [3x3]
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item() #Randomly selects a sample index from the data
                                                                     #len(training_data) gives the total number of samples in the dataset
                                                                     #size=(1,) tells pytorch to return only 1 element tensor
                                                                     #.item() converts the one-element tensor to a plain Python integer.
    img, label = training_data[sample_idx] #Retrieves the image and its corresponding label using the random index.
                                           #Each sample in training_data is a tuple of: (image_tensor, label)
                                           #image_tensor is a 28×28 grayscale image, stored as a PyTorch tensor (shape [1, 28, 28] after ToTensor()).
    figure.add_subplot(rows, cols, i) #Adds a subplot (one of the small images in the grid) to the main figure.
    plt.title(labels_map[label]) #Sets the title above the current subplot using the label from labels_map.
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")  #img.squeeze() removes the channel dimension from the tensor (from [1, 28, 28] to [28, 28]).  im
                                            #img.shape → [1, 28, 28] where img = [channels, height, width] , img.squeeze().shape → [28, 28]
plt.show()


class CustomImageDataset(Dataset): #CustomImageDataset is a child of Dataset which is a class in PyTorch
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None): #Initializes the dataset (self, Path to the CSV file containing image filenames and labels, path to folder containing images,
                                                                                          # Optional image transformation function (e.g., resizing, tensor conversion), Optional transformation function for the labels (e.g., one-hot encoding) )
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels) #Number of samples

    def __getitem__(self, idx): #Getting single data sample(image and label) by index
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]) #os.path.join - joins both the path to get the particular file in the folder
                                                                            #self.img_dir - path to image folder
                                                                            #self.image_labels.iloc[idx,0] - Get the value from row idx, column 0” of the DataFrame. This retrieves the filename of the image at index idx.
        image = decode_image(img_path) #decode_image(path) reads the image as a tensor (typically C×H×W).
        label = self.img_labels.iloc[idx, 1] #Retrieves the label(output/y-value) of the particular image
        if self.transform:
            image = self.transform(image) #could normalize, resize, or augment the image.
        if self.target_transform:
            label = self.target_transform(label) # could convert label into tensor, one-hot vector, etc.
        return image, label

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True) #Dataloader divides the training data into batches
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True) #Dataloader divides the testing data into batches

    # Display image and label.
    train_features, train_labels = next(iter(train_dataloader)) #iter - creates an iterator over the training dataset
                                                                #next -  gives you the first batch from the iterator. Eg. [64, 1, 28, 28]
    print(f"Feature batch shape: {train_features.size()}") #Prints the shape of the feature tensor.
    print(f"Labels batch shape: {train_labels.size()}") #Prints the shape of the label tensor.
    img = train_features[0].squeeze() # takes the first image from the subset and .squeeze() removes the channel
    label = train_labels[0] #Gets the label (class index) for the first image in the batch.
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {labels_map[label.item()]}")