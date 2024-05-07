import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = np.loadtxt('data.csv', delimiter = ',', skiprows = 1)
rows, cols = np.shape(data)

print('Broj izmjerenih ljudi: ', rows)

height = data[:,1]
weight = data[:,2]
plt.scatter(height, weight, s = 1)
plt.xlabel('height')
plt.ylabel('weight')
plt.show()

height50 = height[: : 50]
weight50 = weight[: : 50]
plt.scatter(height50, weight50, s = 5)
plt.xlabel('height')
plt.ylabel('weight')
plt.title('Every 50th person')
plt.show()

print('Minimal height', np.min(height))
print('Maximal height', np.max(height))
print('Average height', np.mean(height))

men = data[np.where(data[:, 0] == 1)]
women = data[np.where(data[:, 0] == 0)]
print('Minimal men height', np.min(men[:, 1]))
print('Maximal men height', np.max(men[:, 1]))
print('Average men height', np.mean(men[:, 1]))
print('Minimal women height', np.min(women[:, 1]))
print('Maximal women height', np.max(women[:, 1]))
print('Average women height', np.mean(women[:, 1]))