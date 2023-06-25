# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import numpy as np
    
class FNN(nn.Module):
    def __init__(self, vocab_size, input_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.norm(x)
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        x = torch.matmul(x, self.embedding.weight.T)
        return x
    

class RNN(nn.Module):
    def __init__(self, vocab_size, input_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.linear_relu = nn.Sequential(
            nn.Linear(hidden_size, embedding_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.norm(x)
        x, _ = self.rnn(x)
        x = self.linear_relu(x)
        x = torch.matmul(x, self.embedding.weight.T)
        return x


class LSTM(nn.Module):
    def __init__(self, vocab_size, input_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear_relu = nn.Sequential(
            nn.Linear(hidden_size, embedding_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.norm(x)
        x, _ = self.lstm(x)
        x = self.linear_relu(x)
        x = torch.matmul(x, self.embedding.weight.T)
        return x


class Trainer():
    def __init__(self, learning_rate, model):
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

    def train(self, dataloader, model=None, loss_fn=None, learning_rate=None):
        if model is None:
            model = self.model
        if loss_fn is None:
            loss_fn = self.loss_fn
        if learning_rate is None:
            optimizer = self.optimizer
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)

            # Compute prediction error
            pred = model(X)
            # print(pred.shape, y.shape)
            loss = loss_fn(pred.reshape(-1,pred.shape[-1]), y.flatten())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch+1) % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(self, dataloader, model=None, loss_fn=None):
        if model is None:
            model = self.model
        if loss_fn is None:
            loss_fn = self.loss_fn
        
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = model(X)
                test_loss += loss_fn(pred.reshape(-1,pred.shape[-1]), y.flatten()).item()
                # print(np.prod(pred.shape[:-1]))
                correct += (pred.argmax(-1) == y).type(torch.float).sum().item()*pred.shape[0]/np.prod(pred.shape[:-1])
        test_loss /= num_batches
        correct /= size
        print(f"Test: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return correct, test_loss