#1 ) Design model (input, output size, forward pass)
#2 ) Construct loss and optimizer
#3 ) Training loop
#    - forward pass: compute prediction
#    - backward pass: gradients
#    - update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 0 ) Prepare data
X_numpy, Y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
Y = torch.from_numpy(Y_numpy.astype(np.float32))
Y = Y.view(Y.shape[0], 1)

n_samples, n_features = X.shape
print(n_samples, n_features)
# 1) model
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

# 2) loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) training loop
num_epochs = 100
for epoch in range(num_epochs):
    # forward pass and loss
    y_predicted = model(X)
    loss = criterion(y_predicted, Y)

    # backward pass
    loss.backward()

    # update
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

# plot
predicted = model(X).detach().numpy()
plt.plot(X_numpy, Y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()


#######################################################################
# # 生成回归数据
# X_numpy, Y_numpy = make_regression(
#     n_samples=100,
#     n_features=1,
#     noise=20,
#     random_state=1
# )

# # 可视化数据
# plt.scatter(X_numpy, Y_numpy, color='blue', label='Data Points')

# # 训练线性回归模型
# model = LinearRegression()
# model.fit(X_numpy, Y_numpy)

# # 预测值
# Y_pred = model.predict(X_numpy)

# # 绘制拟合直线
# plt.plot(X_numpy, Y_pred, color='red', label='Linear Fit')

# # 添加标题和标签
# plt.title('Generated Regression Data with Linear Fit')
# plt.xlabel('Feature (X)')
# plt.ylabel('Target (Y)')
# plt.legend()

# # 显示图形
# plt.show()