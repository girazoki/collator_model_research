from scipy.stats import pareto
import matplotlib.pyplot as plt
import numpy as np
import copy
from random import randint
import argparse

def attacker_minimal_stake(k, leader_stake, sybil_parameter, cost_reduction, leader_cost, total_reward):
  return ((k/2)*(leader_stake-(leader_cost/total_reward*(1-cost_reduction)*(1+(1/sybil_parameter)))))

parser = argparse.ArgumentParser(description='Calculate stake probability')
parser.add_argument('--active_collators', dest='active_collators', type=int, default=max,
                    help='number of active collators', required=True)
parser.add_argument('--sybil_steps', dest='sybil_steps', default=max, type=float,
                    help='sybil parameter steps', required=True)
parser.add_argument('--cost_reduction_steps', dest='cost_reduction_steps', type=float,
                    help='cost reduction steps to consider', required=True) 
parser.add_argument('--leader_cost', dest='leader_cost', type=float,
                    help='leader_cost parameter to consider', required=True) 
parser.add_argument('--leader_stake', dest='leader_stake', type=float,
                    help='leader stake to consider', required=True) 
parser.add_argument('--total_reward', dest='total_reward', type=float,
                    help='total reward to consider', required=True)

args = parser.parse_args()
sybil_steps = np.arange(args.sybil_steps, 1, args.sybil_steps)
cost_reduction_steps = np.arange(args.cost_reduction_steps, 1, args.cost_reduction_steps)
results = np.zeros((len(cost_reduction_steps), len(sybil_steps)))

for i, cost_red in enumerate(cost_reduction_steps):
    for j, sybil in enumerate(sybil_steps):
        results[i][j] = attacker_minimal_stake(args.active_collators, args.leader_stake, sybil, 1-cost_red, args.leader_cost, args.total_reward)

labels = []
for cost_red in cost_reduction_steps:
    labels.append(str("%.2f" % (cost_red*100)) +  ('% cost reduction'))

for y_arr, label in zip(results, labels):
    plt.plot(sybil_steps, y_arr, label=label)

plt.xlabel("Sybil parameter")
plt.ylabel("Minimal Stake for attacker to control half of the active collator set")
plt.legend()
plt.show()