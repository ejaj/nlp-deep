import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


# Define custom Dataset class
class SentimentDataset(Dataset):
    def __init__(self, root_dir, vocab=None, tokenizer=None):
        self.texts = []
        self.labels = []
        self.tokenizer = tokenizer or get_tokenizer('basic_english')
        self.label_map = {'neg': 0, 'pos': 1}  # Map folder names to labels

        # Traverse 'neg' and 'pos' folders
        for label in ['neg', 'pos']:
            folder_path = os.path.join(root_dir, label)
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                with open(file_path, 'r', encoding='utf-8') as file:
                    self.texts.append(file.read())
                    self.labels.append(self.label_map[label])

        # Build vocabulary if not provided
        if vocab is None:
            self.vocab = build_vocab_from_iterator(
                (self.tokenizer(text) for text in self.texts), specials=["<unk>"]
            )
            self.vocab.set_default_index(self.vocab["<unk>"])
        else:
            self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokenized_text = self.tokenizer(self.texts[idx])
        indexed_text = torch.tensor([self.vocab[token] for token in tokenized_text], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return indexed_text, label


# Collate function for padding
def collate_fn(batch):
    texts, labels = zip(*batch)
    texts = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.float32)
    return texts, labels


# Paths
base_dir = '/home/kazi/Works/Projects/nlp-deep/data/review'  # Replace with your actual base folder path
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Load datasets
train_dataset = SentimentDataset(train_dir)
test_dataset = SentimentDataset(test_dir, vocab=train_dataset.vocab)

# DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)


# print(len(train_loader))
# Define RNN model
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(SentimentRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, 60)
        self.fc2 = nn.Linear(60, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.rnn(x)
        x = self.tanh(self.fc1(output[:, -1, :]))
        x = self.fc2(x)
        return x


# Model setup
vocab_size = len(train_dataset.vocab)
embed_size = 64
hidden_size = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentimentRNN(vocab_size, embed_size, hidden_size).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)


# Training loop
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss, total_acc = 0, 0
    for texts, labels in train_loader:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(texts).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += ((torch.sigmoid(outputs) > 0.5) == labels).sum().item()
    return total_loss / len(train_loader), total_acc / len(train_loader.dataset)


# Evaluation loop
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss, total_acc = 0, 0
    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts).squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_acc += ((torch.sigmoid(outputs) > 0.5) == labels).sum().item()
    return total_loss / len(test_loader), total_acc / len(test_loader.dataset)


# Training and evaluation
num_epochs = 10
for epoch in range(num_epochs):
    train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
    print(
        f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
