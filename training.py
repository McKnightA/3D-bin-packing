import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from utils import rollout, discount_rewards, mimic_quanqing_boxes, eval_quanqing_boxes
from environment import ThreeDimBin
from model import PackingModel
from optimizer import PPOTrainer

# Setup
LENGTH = 100
WIDTH = 100
HEIGHT = 500
n_objects = 30
objects = mimic_quanqing_boxes(n_objects, LENGTH, WIDTH)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
n_samples = 60
exp_title = "{} matching quanqing, now with fixed eval".format(n_samples, n_objects)
n_episodes = 1000
print_freq = 10
batch_size = 32

env = ThreeDimBin(LENGTH, WIDTH, HEIGHT, objects)

model = PackingModel(num_samples=n_samples, embedding_dim=128, num_heads=8, num_layers=4)
if os.path.isfile("models/model.pt"):
    model = torch.load("models/model.pt")
model = model.to(DEVICE)

ppo = PPOTrainer(model,
                 policy_lr=3e-4,
                 value_lr=1e-3,
                 target_kl_div=0.005,
                 max_policy_train_iters=60,  # current best at 60
                 value_train_iters=60)

# Training loop
ep_times = []
ep_rewards = []
ep_density = []
ep_remaining_objs = []
for episode_idx in range(n_episodes):
    start_time = time.time()
    # Perform rollout
    train_data, reward = rollout(model, env, objects=mimic_quanqing_boxes(n_objects, LENGTH, WIDTH), device=DEVICE)
    ep_rewards.append(reward)
    ep_density.append(env.calc_reward())

    # record the number of times the blocks are not all placed
    unplaced_obj_count = 0
    for obj in env.remaining_objects:
        if obj is not None:
            unplaced_obj_count += 1
    ep_remaining_objs.append(unplaced_obj_count)

    # Shuffle
    permute_idxs = np.random.permutation(len(train_data[0]))


    def adjust_data_and_train(indeces):
        # Policy data
        obs = []  # the training method is expecting a list of batched bin and object states
        for j in range(len(train_data[0][0])):
            batch = []
            for ind in indeces:
                batch.append(train_data[0][ind][j])
            obs.append(torch.stack(batch, dim=1).squeeze(0).to(DEVICE))

        acts = torch.tensor(train_data[1][indeces], device=DEVICE)
        gaes = torch.tensor(train_data[3][indeces], device=DEVICE)
        act_log_probs = torch.tensor(train_data[4][indeces], device=DEVICE)

        # Value data
        returns = discount_rewards(train_data[2])[indeces]
        returns = torch.tensor(returns, device=DEVICE)

        # Train model
        ppo.train_policy(obs, acts, act_log_probs, gaes, (LENGTH, WIDTH, HEIGHT), DEVICE)
        ppo.train_value(obs, returns)


    if train_data[1].shape[0] > batch_size:
        for i in range(batch_size, train_data[1].shape[0], batch_size):
            adjust_data_and_train(permute_idxs[i - batch_size:i])
    else:
        adjust_data_and_train(permute_idxs)

    ep_times.append(time.time() - start_time)

    # perform validation on new shapes
    score, remain = eval_quanqing_boxes(model, env, n_objects, DEVICE)

    # save the model that does the best packing of all the object
    with open('best_density_yet.txt', 'r') as f:
        best_yet = f.readline()
    if max(remain) == 0 and score >= float(best_yet):
        torch.save(model, "models/model.pt")
        with open('best_density_yet.txt', 'w') as f:
            f.write("{}".format(ep_density[-1]))
        print("saved model")

    if (episode_idx + 1) % print_freq == 0:
        print('Episode {} | Avg Reward {:.2f} | Avg Density {:.2f} | Avg Run Time {:.2f} '
              '| Num Objects Unpacked {}'.format(episode_idx + 1, np.mean(ep_rewards[-print_freq:]),
                                                 np.mean(ep_density[-print_freq:]), np.mean(ep_times[-print_freq:]),
                                                 ep_remaining_objs[-print_freq:]))

        # stop early if already averaging the maximum
        if np.mean(ep_density[-print_freq:]) == 1:
            break

# PLOTS
plt.plot(ep_density)
plt.title("{}".format(exp_title))
plt.ylabel("packing density")
plt.savefig("results/density {}.png".format(exp_title))

plt.plot(ep_rewards)
plt.title("{}".format(exp_title))
plt.ylabel("packing reward")
plt.savefig("results/reward {}.png".format(exp_title))

plt.plot(ep_times)
plt.title("{}".format(exp_title))
plt.ylabel("packing time")
plt.savefig("results/time {}.png".format(exp_title))
