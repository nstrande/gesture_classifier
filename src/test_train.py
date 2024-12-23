import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import json
import os
from pathlib import Path
import numpy as np

class SignLanguageDataset(Dataset):
    def __init__(self, data_dir):
        self.sequences = []
        self.labels = []
        
        # Gennemgå alle gesture mapper
        for gesture_dir in Path(data_dir).iterdir():
            if gesture_dir.is_dir():
                label = gesture_dir.name
                # Load alle sekvenser for dette tegn
                for seq_file in gesture_dir.glob("*.json"):
                    sequence = self.load_sequence(seq_file)
                    self.sequences.append(sequence)
                    self.labels.append(label)
        
        # Konverter labels til numeriske værdier
        self.label_to_idx = {label: i for i, label in enumerate(sorted(set(self.labels)))}
        self.labels = [self.label_to_idx[label] for label in self.labels]
    
    def load_sequence(self, json_path):
        with open(json_path, 'r') as f:
            sequence = json.load(f)
        
        # Find min og max for hver koordinat i denne sekvens
        all_x = [lm['x'] for frame in sequence for lm in frame]
        all_y = [lm['y'] for frame in sequence for lm in frame]
        all_z = [lm['z'] for frame in sequence for lm in frame]
        
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        min_z, max_z = min(all_z), max(all_z)
        
        # Normaliser hver frame
        frames = []
        for frame in sequence:
            landmarks = []
            for landmark in frame:
                landmarks.extend([
                    (landmark['x'] - min_x) / (max_x - min_x) if max_x != min_x else 0,
                    (landmark['y'] - min_y) / (max_y - min_y) if max_y != min_y else 0,
                    (landmark['z'] - min_z) / (max_z - min_z) if max_z != min_z else 0
                ])
            frames.append(landmarks)
            
        return torch.FloatTensor(frames)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def collate_fn(batch):
    # Sorter efter sekvens længde
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    sequences, labels = zip(*batch)
    
    # Pad sekvenser
    lengths = torch.LongTensor([len(seq) for seq in sequences])
    padded_seqs = pad_sequence(sequences, batch_first=True)
    
    return padded_seqs, torch.LongTensor(labels), lengths

class GestureRecognizer(nn.Module):
    def __init__(self, input_dim=63, hidden_dim=128, num_layers=2, num_classes=10, dropout=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x, lengths):
        # Pack sequence
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        
        # LSTM forward pass
        packed_output, _ = self.lstm(packed_x)
        
        # Unpack output
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Attention
        attention_weights = []
        for i, length in enumerate(lengths):
            seq_attention = self.attention(output[i, :length])
            padded_attention = torch.cat([
                seq_attention,
                torch.full((output.size(1) - length, 1), float('-inf'), device=seq_attention.device)
            ])
            attention_weights.append(padded_attention)
        
        attention_weights = torch.stack(attention_weights)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Apply attention
        context = torch.bmm(attention_weights.transpose(1, 2), output)
        context = context.squeeze(1)
        
        # Classification
        output = self.fc(context)
        return output

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, (data, target, lengths) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            lengths = lengths.to(device)
            
            optimizer.zero_grad()
            output = model(data, lengths)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target, lengths in val_loader:
                data, target = data.to(device), target.to(device)
                lengths = lengths.to(device)
                
                output = model(data, lengths)
                val_loss += criterion(output, target).item()
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        val_accuracy = 100 * correct / total
        scheduler.step(val_loss)
        
        # Gem bedste model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f'Epoch: {epoch+1}')
        print(f'Training Loss: {train_loss/len(train_loader):.4f}')
        print(f'Validation Loss: {val_loss/len(val_loader):.4f}')
        print(f'Validation Accuracy: {val_accuracy:.2f}%')
        print('-' * 50)

def main():
    # Setup data
    data_dir = "data/processed"
    dataset = SignLanguageDataset(data_dir)
    
    # Print dataset info
    print(f"Dataset size: {len(dataset)}")
    print(f"Classes: {dataset.label_to_idx}")
    
    # Split data
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)
    
    # Initialize model
    num_classes = len(dataset.label_to_idx)
    model = GestureRecognizer(num_classes=num_classes)
    
    # Train
    train_model(model, train_loader, val_loader)

if __name__ == "__main__":
    main()