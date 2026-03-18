import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Dataset XOR: [[0,0]->0, [0,1]->1, [1,0]->1, [1,1]->0]
X = torch.tensor([[0.0, 0.0], 
                  [0.0, 1.0], 
                  [1.0, 0.0], 
                  [1.0, 1.0]], dtype=torch.float32)

y = torch.tensor([[0.0], 
                  [1.0], 
                  [1.0], 
                  [0.0]], dtype=torch.float32)

class XOR_MLP(nn.Module):
    def __init__(self):
        super(XOR_MLP, self).__init__()
        self.hidden = nn.Linear(2, 4)
        self.output = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU() 

    def forward(self, x):
        x = self.sigmoid(self.hidden(x))
        x = self.sigmoid(self.output(x)) # na wyjsciu wynik 0/1
        return x

model = XOR_MLP()
criterion = nn.BCELoss() # Binary Cross Entropy 
optimizer = optim.SGD(model.parameters(), lr=0.1) 

epochs = 5000 
print("Rozpoczynamy trening...")
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward() # propagacja wsteczna
    optimizer.step() # aktualizacja wag

    if (epoch + 1) % 1000 == 0:
        print(f'Epoka [{epoch+1}/{epochs}], Strata (Loss): {loss.item():.4f}')

print("\nTestowanie wytrenowanego modelu:")
with torch.no_grad(): # Wyłączamy obliczanie gradientów dla fazy testów
    predictions = model(X)
    predicted_classes = (predictions >= 0.5).float() # tu 0.5 zeby uzyskac wyniki 0 i 1
    
    for i in range(len(X)):
        input_data = X[i].tolist()
        expected = y[i].item()
        predicted_val = predictions[i].item()
        predicted_class = predicted_classes[i].item()
        
        print(f"Wejście: {input_data} | Oczekiwane: {expected} | "
              f"Predykcja (wartość): {predicted_val:.4f} -> Klasa: {predicted_class}")



x_min, x_max = -0.5, 1.5 #zakres wartyosci
y_min, y_max = -0.5, 1.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),#100 na 100 punktow
                     np.linspace(y_min, y_max, 100))

# przekształcamy siatkę na tensory (łączymy współrzędne w pary [x, y])
grid = np.c_[xx.ravel(), yy.ravel()]
grid_tensor = torch.tensor(grid, dtype=torch.float32)

with torch.no_grad():
    Z = model(grid_tensor)
    # zmieniamy kształt wyniku z powrotem na siatkę 100x100
    Z = Z.reshape(xx.shape).numpy()


plt.figure(figsize=(8, 6))
# pokolorowane tło decyzyjne z poziomem odcięcia 0.5
plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], cmap='RdBu', alpha=0.7) 

# oryginalne, 4 punkty ze zbioru XOR
plt.scatter(X[:, 0], X[:, 1], c=y.squeeze(), cmap='RdBu', edgecolors='k', s=150)

plt.title("Granica decyzyjna MLP dla problemu XOR")
plt.xlabel("Wejście 1")
plt.ylabel("Wejście 2")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()