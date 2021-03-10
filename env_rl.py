import numpy as np
import matplotlib.pyplot as plt


np.random.seed(0)


width = 15
height = 5
x_final = height - 1
y_final = width - 1
y_wells = [0, 1, 3, 5, 5, 7, 9, 11, 12, 14]
x_wells = [3, 1, 2, 0, 4, 1, 3, 2, 4, 1]
standard_reward = -0.1
tunnel_rewards = np.ones(shape=[height, width]) * standard_reward
for x_well, y_well in zip(x_wells, y_wells):
    tunnel_rewards[x_well, y_well] = -5.0
    tunnel_rewards[x_final, y_final] = 5.0


# fig, ax = plt.subplots()
# ax.matshow(tunnel_rewards, cmap=plt.cm.Blues)
# # show values
# for r in range(tunnel_rewards.shape[0]):
#     for c in range(tunnel_rewards.shape[1]):
#         value = tunnel_rewards[r, c]
#         ax.text(c, r, str(value), va='center', ha='center')
# plt.xlabel('y')
# plt.ylabel('x')
# plt.xticks(np.arange(15))
# plt.show()
# plt.clf()


# gamma = 0.9
nb_actions = 4
policy = np.random.randint(0, nb_actions, size=[height, width]).astype(np.uint8)
tunnel_values = np.zeros(shape=[height, width])


def is_final(x, y):
    if (x, y) in zip(x_wells, y_wells) or (x, y) == (x_final, y_final):
        return True
    return False


def policy_evaluation():
    old_tunnel_values = tunnel_values.copy()
    x, y = 0, 0
    for i in range(height):
        for j in range(width):
            action = policy[i, j]
            if action == 0:
                if i == 0:
                    x = 0
                else:
                    x = i - 1
                    y = j
            elif action == 1:
                if j == width - 1:
                    y = width - 1
                else:
                    x = i
                    y = j + 1
            elif action == 2:
                if i == height - 1:
                    x = height - 1
                else:
                    x = i + 1
                    y = j
            elif action == 3:
                if j == 0:
                    y = 0
                else:
                    y = j - 1
                    x = i
            else:
                Exception('Value out of range')
            reward = tunnel_rewards[x, y]
            tunnel_values[i, j] = reward + gamma * old_tunnel_values[x, y]


def policy_improvement():
    for i in range(height):
        for j in range(width):
            if is_final(i, j):
                continue
            values = np.zeros(shape=[nb_actions, ])

            values[0] = (tunnel_rewards[i - 1, j] + gamma * tunnel_values[i - 1, j]) \
                if i > 0 else -np.inf
            values[1] = (tunnel_rewards[i, j + 1] + gamma * tunnel_values[i, j + 1]) \
                if j < width - 1 else -np.inf
            values[2] = (tunnel_rewards[i + 1, j] + gamma * tunnel_values[i + 1, j]) \
                if i < height - 1 else -np.inf
            values[3] = (tunnel_rewards[i, j - 1] + gamma * tunnel_values[i, j - 1]) \
                if j > 0 else -np.inf

            policy[i, j] = np.argmax(values).astype(np.uint8)


nb_max_epochs = 100_000
tolerance = 1e-5
e = 0
gamma = 0.85
old_policy = np.random.randint(0, nb_actions, size=[height, width]).astype(np.uint8)
while e < nb_max_epochs:
    e += 1
    old_tunnel_values = tunnel_values.copy()
    policy_evaluation()
    if np.mean(np.abs(tunnel_values - old_tunnel_values)) < tolerance:
        old_policy = policy.copy()
    policy_improvement()
    if np.sum(policy - old_policy) == 0:
        break
    print(e)
print('finished')

# TODO recheck










































































































































































































































































































