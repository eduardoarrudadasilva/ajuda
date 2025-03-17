import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Carregar os dados
astrojunior_data = pd.read_csv("treino.csv")

# Definir os inputs (colunas 1 a 13) e os targets (coluna 14)
x_numpy = astrojunior_data.iloc[:, 1:14].values.astype(np.float32)
y_numpy = astrojunior_data.iloc[:, 14].values.astype(np.int64)

# Converter para tensores do PyTorch
x = torch.from_numpy(x_numpy)  # Shape: (n_amostras, 13)
y = torch.from_numpy(y_numpy)  # Shape: (n_amostras,)

# Definição da classe do modelo
class RegressaoSoftmax(nn.Module):
    def __init__(self, n_input, n_output):
        super(RegressaoSoftmax, self).__init__()
        self.Linear = nn.Linear(n_input, n_output)

    def forward(self, x):
        return self.Linear(x)  # Softmax NÃO deve ser aplicado aqui!

# Definição dos tamanhos de entrada e saída
input_size = x.shape[1]  # Deve ser 13
output_size = 5  # Número de classes na classificação
model = RegressaoSoftmax(input_size, output_size)

# Definição da função de custo e otimizador
learning_rate = 0.05
criterion = nn.CrossEntropyLoss()  # Já inclui Softmax internamente
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Loop de treinamento
num_epochs = 1000
contador_custo = []

for epoch in range(num_epochs):
    # Forward pass
    y_hat = model(x)  # A saída bruta (logits), sem Softmax
    loss = criterion(y_hat, y)  # CrossEntropyLoss já faz Softmax

    # Armazenar a perda corretamente
    contador_custo.append(loss.item())  # .item() para pegar o valor numérico

    # Backward pass (calcular gradientes)
    loss.backward()

    # Atualizar os pesos
    optimizer.step()

    # Limpar os gradientes acumulados
    optimizer.zero_grad()

# Plotando o gráfico da função de custo
print("GRÁFICO DA FUNÇÃO DE CUSTO")
plt.plot(contador_custo, 'b')
plt.xlabel("Épocas")
plt.ylabel("Perda (Loss)")
plt.title("Evolução da Função de Custo")
plt.show()
