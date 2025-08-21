import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader


# # Removing the first 4 columns that are not required

# df = pd.read_csv("matched_Landmark.csv",header=None)
# df_cropped = df.iloc[:,4:]
# df_cropped.to_csv("Landmark.csv")
#
# df2 = pd.read_csv("matched_target.csv",header=None)
# df_cropped2 = df2.iloc[:,4:]
# df_cropped2.to_csv("Target.csv")




# Inverting the Table
landmark = pd.read_csv("Landmark.csv",header=None)
# print(landmark.iloc[1:,1:])
final_landmark = landmark.iloc[1:,1:].T # We remove the first column and row which contains indexing
# print(final_landmark)

Target = pd.read_csv("Target.csv",header=None)
# print(Target)
final_Target = Target.iloc[1:,1:].T # We remove the first column and row which contains indexing
# print(final_Target)



# Train_Test_Split
X_train , X_test, y_train, y_test = train_test_split(final_landmark,final_Target,train_size=0.7,random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,train_size=0.9,random_state=42)


# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # fit on train
X_val   = scaler.transform(X_val)         # transform val
X_test  = scaler.transform(X_test)        # transform test


y_train = scaler.fit_transform(y_train)
y_val = scaler.transform(y_val)
y_test = scaler.transform(y_test)


# Converting to a Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32) # dtype=torch.long for Single Label Classification
y_train_tensor = torch.tensor(y_train, dtype=torch.float32) # y_train.values if it is not a numpy array/matrix (ie without standardization)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


print("Train:", X_train_tensor.shape)
print("Train:", y_train_tensor.shape)
print("Val:", X_val_tensor.shape)
print("Test:", X_test_tensor.shape)



# Creating Datasets and DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor) # Creating custom dataset for all three train, val and test
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # Loading the data to the dataloader
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)


# Creating the Architecture
class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.LinearFFN = nn.Sequential (
            nn.Linear(942,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024,9520)

        )

    def forward(self,x):
        logits = self.LinearFFN(x)
        return logits

model = FFN()


# ---------Setting the loss and optimizer-----------
# loss_fn = nn.CrossEntropyLoss() # # Single Label Classification
# loss_fn = nn.BCEWithLogitsLoss() # Multi Class Classification
loss_fn = nn.MSELoss()   # mean squared error
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 10

# Training and Validation
for epoch in range(num_epochs):
    # Training the model
    model.train() # Sets the model to training mode (Used when dropout and/or BatchNorm is present)
    train_loss = 0 # Keeps track of the total training loss for the epoch.
    for X_train, y_train in train_loader:
        optimizer.zero_grad() # Clears gradients from the previous iteration
        outputs = model(X_train) # Running the FFN on the Training data
        loss = loss_fn(outputs, y_train) # Compute the loss
        loss.backward() # Back Propagation
        optimizer.step() # Updates the model parameters (weights) using the chosen optimizer
        train_loss += loss.item() # Adds the batch loss into train loss

    # Validation using the Val Set
    model.eval() # Sets the models for testing mode
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch.squeeze())
            val_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}] "
          f"Train Loss: {train_loss / len(train_loader):.4f} " # We divide by the number of samples because
          f"Val Loss: {val_loss / len(val_loader):.4f} ")


# Testing on the Test Data
model.eval()
test_loss = 0
with torch.no_grad(): # Disables gradient tracking (If not used it can lead to unnecessary memory usage)
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        test_loss += loss_fn(outputs, y_batch).item()

print(f"Test Loss (MSE): {test_loss / len(test_loader):.4f}")






