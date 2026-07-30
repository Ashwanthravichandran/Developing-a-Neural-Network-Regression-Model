# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: ASHWANTH R

### Register Number: 212224040033

```python
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
data = pd.read_csv(
    r"C:\Users\admin\Desktop\DL\exp1\Developing-a-Neural-Network-Regression-Model\Exp-1.csv"
)

print(data.head())
print(data.columns)
x = data[['Input']]
y = data[['Output']]
scale_x = MinMaxScaler()
scale_y = MinMaxScaler()

x_scaled = scale_x.fit_transform(x)
y_scaled = scale_y.fit_transform(y)
X_train, X_test, Y_train, Y_test = train_test_split(
    x_scaled,
    y_scaled,
    test_size=0.2,
    random_state=42
)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)
class NeuralNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.network(x)
model = NeuralNet()

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)
epochs = 1000

losses = []

for epoch in range(epochs):

    optimizer.zero_grad()

    prediction = model(X_train_tensor)

    loss = criterion(prediction, Y_train_tensor)

    loss.backward()

    optimizer.step()

    losses.append(loss.item())

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs}  Loss = {loss.item():.6f}")
model.eval()

with torch.no_grad():

    test_prediction = model(X_test_tensor)

    test_loss = criterion(test_prediction, Y_test_tensor)

print("Test Loss:", test_loss.item())
plt.figure(figsize=(8,5))

plt.plot(losses)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")

plt.grid(True)

plt.show()
# Prediction
sample = pd.DataFrame([[16]], columns=["Input"])

sample_scaled = scale_x.transform(sample)

sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32)

with torch.no_grad():
    prediction = model(sample_tensor)

prediction_actual = scale_y.inverse_transform(prediction.numpy())

print(f"Predicted Output: {prediction_actual[0][0]:.2f}")

```

### Dataset Information


![alt text](<Screenshot 2026-07-30 112414.png>)

### OUTPUT


![alt text](<Screenshot 2026-07-30 112425.png>)





### Training Loss Vs Iteration Plot


![alt text](<Screenshot 2026-07-30 112447.png>)

![alt text](<Screenshot 2026-07-30 112435.png>)


### New Sample Data Prediction



![alt text](<Screenshot 2026-07-30 112455.png>)


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
