# Recidivism Prediction Using Off-Policy PPO and Improved DE

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
import os
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# Reproducibility Setup
# -----------------------------
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Utility: Data Reader
# -----------------------------
def load_data(file_path):
    df = pd.read_csv(file_path)
    y = df["recidivism"]
    X = df.drop(columns=["recidivism"])
    return X, y


# -----------------------------
# Feature Selector and Predictor (MLP)
# -----------------------------
class FeatureSelectorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate, activation_fn):
        super(FeatureSelectorMLP, self).__init__()
        layers = []
        last_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(last_dim, dim))
            layers.append(self.get_activation(activation_fn))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            last_dim = dim
        self.mlp = nn.Sequential(*layers)
        self.feature_score_layer = nn.Linear(last_dim, input_dim)
        self.output_layer = nn.Linear(last_dim, output_dim)

    def get_activation(self, name):
        if name == "relu": return nn.ReLU()
        if name == "leaky_relu": return nn.LeakyReLU()
        if name == "sigmoid": return nn.Sigmoid()
        if name == "tanh": return nn.Tanh()
        return nn.Identity()

    def forward(self, x):
        features = self.mlp(x)
        feature_scores = torch.sigmoid(self.feature_score_layer(features))
        output = torch.sigmoid(self.output_layer(features))
        return feature_scores, output


# -----------------------------
# Off-Policy PPO Class
# -----------------------------
class OffPolicyPPO:
    def __init__(self, policy_net, lr, gamma=0.99, clip_eps=0.2):
        self.policy_net = policy_net
        self.old_policy_net = FeatureSelectorMLP(policy_net.mlp[0].in_features, [layer.out_features for layer in policy_net.mlp if isinstance(layer, nn.Linear)], 1, 0, "relu")
        self.old_policy_net.load_state_dict(policy_net.state_dict())
        self.optimizer = optim.Adam(policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_eps = clip_eps

    def update(self, memory):
        states, actions, rewards, next_states = zip(*memory)
        states = torch.stack(states)
        actions = torch.tensor(actions).long()
        rewards = torch.tensor(rewards).float()

        logits = self.policy_net(states)[1].squeeze()
        old_logits = self.old_policy_net(states)[1].squeeze().detach()
        dist = Categorical(logits)
        old_dist = Categorical(old_logits)

        ratios = torch.exp(dist.log_prob(actions) - old_dist.log_prob(actions))
        advantages = rewards - rewards.mean()

        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages

        loss = -torch.min(surr1, surr2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.old_policy_net.load_state_dict(self.policy_net.state_dict())


# -----------------------------
# Improved DE for Hyperparameter Optimization
# -----------------------------
class ImprovedDE:
    def __init__(self, param_space, max_fes=1000, F=0.5, CR=0.7, pop_size=20):
        self.param_space = param_space
        self.max_fes = max_fes
        self.F = F
        self.CR = CR
        self.pop_size = pop_size

    def initialize_population(self):
        population = []
        for _ in range(self.pop_size):
            individual = {}
            for param, config in self.param_space.items():
                if config['type'] == 'int':
                    individual[param] = np.random.randint(config['min'], config['max'] + 1)
                elif config['type'] == 'float':
                    individual[param] = np.random.uniform(config['min'], config['max'])
                elif config['type'] == 'cat':
                    individual[param] = random.choice(config['values'])
            population.append(individual)
        return population

    def fitness(self, individual, X, y):
        model = train_rl_model(X, y, individual)
        # Placeholder for real evaluation, e.g., validation AUC or F1
        return np.random.rand()

    def mutate(self, r1, r2, r3):
        mutant = {}
        for key in r1:
            if isinstance(r1[key], (int, float)):
                val = r1[key] + self.F * (r2[key] - r3[key])
                val = np.clip(val, self.param_space[key]['min'], self.param_space[key]['max'])
                if self.param_space[key]['type'] == 'int':
                    val = int(val)
                mutant[key] = val
            else:
                mutant[key] = random.choice(self.param_space[key]['values'])
        return mutant

    def crossover(self, target, donor):
        return {key: donor[key] if random.random() < self.CR else target[key] for key in target}

    def run(self, X, y):
        population = self.initialize_population()
        fitnesses = [self.fitness(ind, X, y) for ind in population]
        fes = 0

        while fes < self.max_fes:
            new_population = []
            for i in range(self.pop_size):
                indices = list(range(self.pop_size))
                indices.remove(i)
                r1, r2, r3 = random.sample(indices, 3)
                donor = self.mutate(population[r1], population[r2], population[r3])
                trial = self.crossover(population[i], donor)
                trial_fit = self.fitness(trial, X, y)
                fes += 1
                if trial_fit < fitnesses[i]:
                    new_population.append(trial)
                    fitnesses[i] = trial_fit
                else:
                    new_population.append(population[i])
            population = new_population

        best_index = np.argmin(fitnesses)
        return population[best_index]


# -----------------------------
# Main RL Training Loop (Simplified)
# -----------------------------
def train_rl_model(X, y, params):
    input_dim = X.shape[1]
    layers = [params['layer_size']] * params['num_layers']
    model = FeatureSelectorMLP(input_dim, layers, 1, dropout_rate=params['dropout'], activation_fn=params['activation'])
    ppo = OffPolicyPPO(model, lr=params['lr'], clip_eps=params['clip_eps'])
    memory = []
    for epoch in range(params['epochs']):
        for i in range(len(X)):
            x_i = torch.tensor(X.iloc[i].values, dtype=torch.float32)
            y_i = y.iloc[i]
            feature_scores, pred = model(x_i)
            reward = (1 if round(pred.item()) == y_i else -1)
            memory.append((x_i, 0, reward, x_i))
        ppo.update(memory)
    return model


# -----------------------------
# Main Execution Function
# -----------------------------
def main():
    set_seed(42)
    file_path = "data/recidivism_data.csv"  # update with actual path
    X, y = load_data(file_path)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    param_space = {
        'batch_size': {'type': 'int', 'min': 16, 'max': 256},
        'epochs': {'type': 'int', 'min': 32, 'max': 1024},
        'activation': {'type': 'cat', 'values': ['relu', 'leaky_relu', 'sigmoid', 'tanh', 'linear']},
        'dropout': {'type': 'float', 'min': 0.0, 'max': 1.0},
        'num_layers': {'type': 'int', 'min': 1, 'max': 8},
        'clip_eps': {'type': 'float', 'min': 0.1, 'max': 0.3},
        'lr': {'type': 'float', 'min': 0.0001, 'max': 0.1},
        'gamma': {'type': 'float', 'min': 0.0, 'max': 1.0},
        'layer_size': {'type': 'int', 'min': 16, 'max': 256}
    }

    de_optimizer = ImprovedDE(param_space, max_fes=50, pop_size=10)
    best_params = de_optimizer.run(X, y)
    print("Best hyperparameters found:", best_params)

    final_model = train_rl_model(X, y, best_params)
    print("Training completed with optimized hyperparameters.")


if __name__ == '__main__':
    main()
