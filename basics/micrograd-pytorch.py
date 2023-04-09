import torch
import torch.nn as nn
import torch.optim as optim

# 1. Import required libraries
device = torch.device("cuda" if torch.cuda.is_available() else \
                      ("mps" if torch.backends.mps.is_available() else "cpu"))

# 2. Define the MLP architecture
class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers_size):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_layers_size[0]))
        layers.append(nn.Tanh())
        for i in range(len(hidden_layers_size) - 1):
            layers.append(nn.Linear(hidden_layers_size[i], hidden_layers_size[i + 1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_layers_size[-1], 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# 3. Create a dataset
X = torch.tensor([[1.0, 6.0, 0.0],
                  [0.0, 3.0, 1.0],
                  [2.0, 4.0, 0.0],
                  [0.0, 3.0, 2.0],
                  [3.0, 2.0, 0.0],
                  [0.0, 1.0, 3.0]], dtype=torch.float32).to(device)

y = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=torch.float32).to(device)

# 4. Setup the training loop
mlp = MLP(3, [4, 4]).to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(mlp.parameters(), lr=0.005)

# 5. Train the network
num_epochs = 5000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = mlp(X).squeeze()
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Test the network on the first entry
output = mlp(X[0])
print(f"Output: {output.item()}, Expected: {y[0].item()}")
