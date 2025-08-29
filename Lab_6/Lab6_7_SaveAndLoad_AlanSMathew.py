import torch
import torchvision.models as models

model = models.vgg16(weights='IMAGENET1K_V1') #VGG16 is a DL model and weights are extracted for the model which is trained on Imagenet1k dataset
torch.save(model.state_dict(), 'model_weights.pth')
#model.state_dict() A PyTorch model has two main parts:
    #Architecture (the layers and their connections).
    #Parameters (weights and biases of each layer).
#state_dict() extracts just the parameters (weights and biases), not the architecture itself
#and save to a file called model_weights.pth
#
model = models.vgg16() # we do not specify ``weights``, i.e. create untrained model
model.load_state_dict(torch.load('model_weights.pth', weights_only=True)) # Loads the weights which we saved earlier
model.eval() #Switch model to evaluation mode

torch.save(model, 'model.pth') # Saves the entire model with architecture, weights, bias and other attributes

model = torch.load('model.pth', weights_only=False),