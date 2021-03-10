import numpy as np
import matplotlib.pyplot as plt


width = 15
height = 5
x_final = width - 1
y_final = height - 1
x_wells = [0, 1, 3, 5, 5, 7, 9, 11, 12, 14]
y_wells = [3, 1, 2, 0, 4, 1, 3, 2, 4, 1]
standard_reward = -0.1
tunnel_rewards = np.ones(shape=[height, width]) * standard_reward
for x_well, y_well in zip(y_wells, x_wells):
    tunnel_rewards[y_wells, x_wells] = -5.0
    tunnel_rewards[y_final, x_final] = 5.0


fig, ax = plt.subplots()
ax.matshow(tunnel_rewards, cmap=plt.cm.Blues)
# show values
for r in range(tunnel_rewards.shape[0]):
    for c in range(tunnel_rewards.shape[1]):
        value = tunnel_rewards[r, c]
        ax.text(c, r, str(value), va='center', ha='center')
plt.xlabel('x')
plt.ylabel('y')
plt.xticks(np.arange(15))
plt.show()
plt.clf()



















































































































































































































































































































