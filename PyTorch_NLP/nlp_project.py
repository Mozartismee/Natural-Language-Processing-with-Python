import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data import Field, LabelField, BucketIterator

# Define a GRU-based sentiment analysis model
class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, n_layers, dropout):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers=n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        embed = self.embedding(x)
        _, h_n = self.gru(embed)
        out = self.fc(h_n[-1,:,:])
        return out

# Create fields for text and labels
TEXT = Field(tokenize='spacy', lower=True)
LABEL = LabelField(dtype=torch.float)

# Load the IMDB dataset
train_data, test_data = IMDB.splits(TEXT, LABEL)

# Build the vocabulary using the top 10,000 frequently occurring words
TEXT.build_vocab(train_data, max_size=10000)
LABEL.build_vocab(train_data)

# Set device, hyperparameters, and create iterators
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab_size = len(TEXT.vocab)
embedding_dim = 100
hidden_size = 256
output_size = 1
n_layers = 2
dropout = 0.5
lr = 0.001
n_epochs = 5
batch_size = 64

train_iter, test_iter = BucketIterator.splits(
    (train_data, test_data), batch_size=batch_size, device=device)

# Initialize the model and set the optimizer and loss function
model = SentimentModel(vocab_size, embedding_dim, hidden_size, output_size, n_layers, dropout).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()

# Training loop
for epoch in range(n_epochs):
    train_loss = 0.0
    
    for batch in train_iter:
        optimizer.zero_grad()

        text, label = batch.text, batch.label

        output = model(text).squeeze()

        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Epoch: {epoch+1}, Loss: {train_loss/len(train_iter):.4f}")

# Save the model
torch.save(model.state_dict(), 'sentiment_model.pt')
