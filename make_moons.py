# Import required libraries
import torch
from torch import nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Set device to GPU if available else CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Make data
n_samples = 1000
X, y = make_moons(n_samples=n_samples, noise=0.2, random_state=42)

# Convert data to tensors
X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32)
print(X.shape) # This is the in_features!
# Split data into 80-20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Make model
class MoonModelV0(nn.Module):
    def __init__(self):
        super(MoonModelV0, self).__init__()
        self.network = nn.Sequential(nn.Linear(in_features=2, out_features=16),
                                     nn.ReLU(),
                                     nn.Linear(in_features=16, out_features=64),
                                     nn.ReLU(),
                                     nn.Linear(in_features=64, out_features=64),
                                     nn.ReLU(),
                                     nn.Linear(in_features=64, out_features=16),
                                     nn.ReLU(),
                                     nn.Linear(in_features=16, out_features=1))
    def forward(self, x):
        return self.network(x)

# Accuracy function
def accuracy_fn(y_true, y_pred):
    """Calculate classification accuracy (as percentage)."""
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

# Train model and save weights to file
def train_and_save_model(model, X_train,  X_test, y_train, y_test, loss_fn, optimizer, epochs=3000, model_path="moob_model.pth"):
    X_train, X_test = X_train.to(device), X_test.to(device)
    y_train, y_test = y_train.to(device).unsqueeze(1), y_test.to(device).unsqueeze(1)
    for epoch in range(epochs):
        model.train()
        # Forward pass
        train_logits = model(X_train.squeeze())
        train_pred = torch.round(torch.sigmoid(train_logits))
        # Calculate accuracy and loss
        train_loss = loss_fn(train_logits, y_train)
        train_acc = accuracy_fn(y_train, train_pred)
        # Backpropagation and optimizer step
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        # Evaluate test set
        model.eval()
        with torch.inference_mode():
            test_logits = model(X_test)
            test_pred = torch.round(torch.sigmoid(test_logits))
            test_loss = loss_fn(test_logits, y_test)
            test_acc = accuracy_fn(y_test, test_pred)
        if epoch % 100 == 0:
            print(f"Epoch: {epoch} | Loss: {train_loss:.5f}, Acc: {train_acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved in path: {model_path}")

#Plot decision boundary
def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """Original Code Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    Modified by mrdbourke and taken from the amazing helper_functions.py"""

    #Move everything to CPU for compatibility with numpy/matplotlib
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")
    #Create mesh grid over feature space
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()
    #Get model predictions for each point in the grid
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)
    #Binary classification: apply sigmoid then round
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # For multi-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))          # For binary
    #Reshape and plot decision boundary
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()

def load_and_predict(model_class, model_path, X, y):
    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    with torch.inference_mode():
        test_pred = torch.round(torch.sigmoid(model(X.to(device).squeeze())))
    plot_decision_boundary(model, X_test, y_test)

if __name__ == "__main__":
    # Init model, loss_fn, optimizer
    model = MoonModelV0().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    # Train and save model
    train_and_save_model(model, X_train, X_test, y_train, y_test, loss_fn, optimizer)
    # Load and predict
    load_and_predict(MoonModelV0, "moob_model.pth", X_test, y_test)



