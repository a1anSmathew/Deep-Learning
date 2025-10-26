import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.nn.utils.rnn import pad_sequence

df = pd.read_csv('Flickr8k/captions.txt')
print(df.head(5))

grouped = df.groupby('image')['caption'].apply(list)

rows = []
for img, captions in grouped.items():
    rows.append([img, captions[0]])  #taking first caption

csv_df = pd.DataFrame(rows, columns=['image', 'caption'])

csv_df['image'].to_csv('Flickr8k/Images_only/all_images.csv', index=False, header=False)
csv_df['caption'].to_csv('Flickr8k/Captions_only/all_captions.csv', index=False, header=False)

print("Saved single CSVs for images and captions.")

class EncoderCNN(nn.Module):
    def __init__(self, embed_size=256):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]  #removing last FC layer
        self.resnet = nn.Sequential(*modules) #unpacking
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)  # (batch, 512, 1, 1)
        features = features.view(features.size(0), -1)  # flatten (batch, 512)
        features = self.linear(features)  # (batch, embed_size)
        features = self.bn(features)
        return features

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

encoder = EncoderCNN(embed_size=256)

def extract_features(images_csv, img_folder, save_file, encoder, transform):
    df = pd.read_csv(images_csv, header=None)
    image_names = df[0].to_list()
    features_for_each_image = []
    img_folder = "Flickr8k/Images_only/Images"

    encoder.eval()
    for img in image_names:
        img_path = f'{img_folder}/{img}'
        img = Image.open(img_path).convert('RGB')
        img = transform(img).unsqueeze(0)

        with torch.no_grad():
            features = encoder(img)
            features_for_each_image.append(features)

    #Convert list of features to tensor
    features_tensor = torch.cat(features_for_each_image, dim=0)  # shape: (num_images, embed_size)

    # Save features to file if needed
    torch.save(features_tensor, save_file)
    print(f"Extracted features for {len(features_for_each_image)} images and saved to {save_file}")

    return features_tensor

#load the created csvs

image_csv = 'Flickr8k/Images_only/all_images.csv'
caption_csv = 'Flickr8k/Captions_only/all_captions.csv'

captions = pd.read_csv(caption_csv, header=None)[0].tolist()

split_captions = [c.lower().split() for c in captions]  #split the captions into words.
all_words = [w for cap in split_captions for w in cap]
vocab = {w: i+1 for i, w in enumerate(set(all_words))}  #assign numbers to words of each caption in the list of all split captions.
vocab['<start>'] = len(vocab) #to identify the start and end of a caption (same integer)
vocab['<end>'] = len(vocab)
vocab['<unknown>'] = len(vocab)
inv_vocab = {i: w for w, i in vocab.items()} #now we will have numbers and we want to generate words from it (caption generation). So, we need to do inverse mapping, i.e., numbers to words.
                                            #during training

caption_indices = []
for cap in split_captions:
    idx = [vocab['<start>']] + [vocab.get(w, vocab['<unknown>']) for w in cap] + [vocab['<end>']]
    caption_indices.append(idx)

#for example:
# cap = ['a', 'dog', 'is', 'running']
# idx = [10] + [1, 2, 3, 4] + [10]
# => idx = [10, 1, 2, 3, 4, 10]

#caption_indices = [
#     [10, 1, 2, 3, 4, 10],
#     [10, 5, 6, 7, 8, 9, 10],
#     ...
# ]

#caption generator RNN:
class CaptionGenerator(nn.Module):
    def __init__(self, feature_dim, hidden_dim, vocab_size):
        super(CaptionGenerator, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # Simple embedding layer for words
        self.embed = nn.Embedding(vocab_size, feature_dim)

        # LSTM
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, batch_first=True)

        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, captions=None):
        """
        features: (batch, 1, feature_dim)
        captions: (batch, seq_len) - optional for training
        """
        if captions is not None:
            # Embed caption words (skip last word)
            embeddings = self.embed(captions[:, :-1])  # (batch, seq_len-1, feature_dim)
            # Concatenate image feature as first input
            inputs = torch.cat((features, embeddings), dim=1)  # (batch, seq_len, feature_dim)
        else:
            # Inference: only features initially
            inputs = features

        lstm_out, _ = self.lstm(inputs)
        outputs = self.fc(lstm_out)
        return outputs


feature_dim = 256 #(because in ResNet, embed_size was 256)
hidden_dim = 256
vocab_size = len(vocab)

model = CaptionGenerator(feature_dim=256, hidden_dim=256, vocab_size=len(vocab))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
features = extract_features(
    image_csv,
    "/home/ibab/DL/Flickr8k/Images_only/Images",
    "features.pt",
    encoder,
    transform
)

epochs = 3
model.train()

for epoch in range(epochs):
    total_loss = 0
    for i in range(len(features)):
        image_feature = features[i].unsqueeze(0).unsqueeze(1)  # (1,1,256)
        caption = torch.tensor(caption_indices[i]).unsqueeze(0)  # (1, seq_len)

        optimizer.zero_grad()
        outputs = model(image_feature, caption)  # (1, seq_len, vocab_size)
        outputs = outputs[:, :caption.size(1), :]
        loss = criterion(outputs.reshape(-1, vocab_size), caption[:, :].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(features):.4f}")


max_length = 15 #maximum length of caption it generates to avoid infinite looping.

def generate_caption(model, feature, inv_vocab, max_length=15):
    model.eval()
    caption = ["<start>"]

    feature = feature.unsqueeze(0).unsqueeze(1)  # (1,1,256)
    input_word = torch.tensor([vocab["<start>"]]).unsqueeze(0)  # (1,1)

    hidden = None
    for _ in range(max_length):
        embedding = model.embed(input_word)  # (1,1,256)
        lstm_input = torch.cat((feature, embedding), dim=1)  # (1,2,256)
        lstm_out, hidden = model.lstm(lstm_input, hidden)
        output = model.fc(lstm_out[:, -1, :])  # take last timestep
        predicted_id = torch.argmax(output, dim=1).item()
        word = inv_vocab.get(predicted_id, "<unknown>")

        if word == "<end>":
            break

        caption.append(word)
        input_word = torch.tensor([[predicted_id]])
        feature = torch.zeros_like(feature)  #set to 0

    return " ".join(caption)


sample_feature = features[0]
generated_caption = generate_caption(model, sample_feature, inv_vocab)
print("Generated Caption:", generated_caption)
