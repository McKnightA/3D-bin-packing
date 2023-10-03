import torch
import torch.optim as optim
from utils import get_action_distributions


# implement PPO
class PPOTrainer:
    """
    train an agent using proximal policy optimization
    """

    def __init__(self, actor_critic,
                 ppo_clip_val=0.2,
                 target_kl_div=0.01,
                 max_policy_train_iters=80,
                 value_train_iters=80,
                 policy_lr=3e-4,
                 value_lr=1e-2):
        """

        :param actor_critic: the model that generates action policies and state values based on observations
        :param ppo_clip_val: what range around 1 should policy change ratios be bound between
        :param target_kl_div: if over the training iterations the kl divergence, between new and old distributions,
         changes by more than this value, stop the training loop early
        :param max_policy_train_iters: how many times should the policy network be updated based on this data
        :param value_train_iters: how many times should the value network be updated based on this data
        :param policy_lr: the learning rate for the policy network optimizer
        :param value_lr: the learning rate for the value network optimizer
        """
        self.ac = actor_critic
        self.ppo_clip_val = ppo_clip_val
        self.target_kl_div = target_kl_div
        self.max_policy_train_iters = max_policy_train_iters
        self.value_train_iters = value_train_iters

        pol_params = [item.parameters() for item in self.ac.encoding_layers] + \
                     [item.parameters() for item in self.ac.policy_layers]
        policy_params = []
        for param in pol_params:
            policy_params = policy_params + list(param)
        self.policy_optim = optim.Adam(policy_params, lr=policy_lr)

        val_params = [item.parameters() for item in self.ac.encoding_layers] + \
                     [item.parameters() for item in self.ac.value_layers]
        value_params = []
        for param in val_params:
            value_params = value_params + list(param)
        self.value_optim = optim.Adam(value_params, lr=value_lr)

    def train_policy(self, obs, acts, old_log_probs, advantages, bin_shape, device='cpu'):
        """
        The method to train the action policy network
        :param obs: the observed environment states taking the form of len(list of batched bin and remaining objects)
        :param acts: the actions performed at each step taking the form of (batch, 7)
        :param old_log_probs: the log probabilities of acts with the policy distributions from before training
        :param advantages: the generalized advantage estimates taking the form of ...
        :param bin_shape: the (x, y, z) size of the packing bin
        :param device:
        """

        for _ in range(self.max_policy_train_iters):
            self.policy_optim.zero_grad()

            new_logits = self.ac.get_policy(self.ac.encode(obs))

            # the new log probs are determined by the old action and the new distribution,
            # so I only need to generate the new distribution
            new_distributions = get_action_distributions(new_logits, len(obs) - 1, bin_shape, obs[1:], device=device)
            ind_log_prob = new_distributions[0].log_prob(acts[:, 0])
            pos_x_log_prob = new_distributions[1].log_prob(acts[:, 1])
            pos_y_log_prob = new_distributions[2].log_prob(acts[:, 2])
            pos_z_log_prob = new_distributions[3].log_prob(acts[:, 3])
            rot_x_log_prob = new_distributions[4].log_prob(acts[:, 4])
            rot_y_log_prob = new_distributions[5].log_prob(acts[:, 5])
            rot_z_log_prob = new_distributions[6].log_prob(acts[:, 6])
            new_log_probs = torch.stack([ind_log_prob, pos_x_log_prob, pos_y_log_prob, pos_z_log_prob,
                                         rot_x_log_prob, rot_y_log_prob, rot_z_log_prob], dim=1)

            # each policy has its own ratio w.r.t the corresponding old policy
            policy_ratios = []
            for i in range(new_log_probs.shape[1]):
                policy_ratios.append(torch.exp(new_log_probs[:, i] - old_log_probs[:, i]))

            # average the 7 policy ratios because ppo only expects to deal with one
            policy_ratio = torch.mean(torch.stack(policy_ratios, dim=1), dim=1)
            clipped_ratio = policy_ratio.clamp(1 - self.ppo_clip_val, 1 + self.ppo_clip_val)

            clipped_loss = clipped_ratio * advantages  # derivative is only flowing through advantages, so only the
            full_loss = policy_ratio * advantages
            policy_loss = -torch.min(full_loss, clipped_loss).mean()

            policy_loss.backward()
            self.policy_optim.step()

            # leave the loop early if the policy shift passes a threshold
            kl_div = (old_log_probs - new_log_probs).mean()
            if kl_div >= self.target_kl_div:
                break

    def train_value(self, obs, returns):
        """
        The method to train the state value network
        :param obs: the observed environment states taking the form of (batch, len(list of bin and remaining objects))
        :param returns: the rewards returned by the environment
        """
        for _ in range(self.value_train_iters):
            self.value_optim.zero_grad()

            values = self.ac.get_value(self.ac.encode(obs))
            value_loss = (returns - values) ** 2
            value_loss = value_loss.mean()

            value_loss.backward()
            self.value_optim.step()
