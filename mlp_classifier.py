import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=[64, 32]):
        super().__init__()
        in_dim = input_dim

        self.hidden_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for h_dim in hidden_dims:
            self.hidden_layers.append(nn.Linear(in_dim, h_dim))
            self.batch_norms.append(nn.BatchNorm1d(h_dim))
            self.activations.append(nn.LeakyReLU(negative_slope=0.01))
            self.dropouts.append(nn.Dropout(p=0.2, ))  
            in_dim = h_dim

        self.out = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        for linear, bn, act, do in zip(self.hidden_layers, self.batch_norms, self.activations, self.dropouts):
            x = linear(x)
            x = bn(x)
            x = act(x)
            x = do(x)
        logits = self.out(x)
        return logits
    
def train_mlp(model, dataloader, criterion, num_epochs=10, lr=1e-4, weight_decay=1e-5):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    y_true_all = []
    y_pred_all = []

    for epoch in range(num_epochs):
        losses = []
        for x_batch, y_batch in dataloader:
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            y_true_all.append(y_batch.detach().cpu())
            y_pred_all.append(logits.detach().cpu())

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {np.mean(losses):.4f}')


def predict_mlp(model, dataloader):
    model.eval()
    all_logits = []
    with torch.no_grad():
        for x_batch, _ in dataloader:
            logits = model(x_batch)
            all_logits.append(logits)
    return torch.cat(all_logits, dim=0)