import argparse
import numpy as np
import matplotlib.pyplot as plt
import math

from scipy.stats import binom, binom_test
import scipy.special

n = 0 #number of collators
alpha = 0 #honest collators
c = 0 #difficulty parameter


#finds the probability of only honest validators are selected
def prH(c, alpha):
    p = 1 - pow(1-c,1/n) #the probability of being a slot leader for a validator
    m = int(alpha * n)
    b = binom(m,p)
    sum = 0
    for k in range(1,m+1):
        sum = sum + b.pmf(k)
    return pow(1-c, 1-alpha) * sum
    

#finds the probabolity that at least one validator is selected
def prh(c, alpha):
    return c - prM(c, alpha)

#finds the probability that only malicious validators are seleceted
def prM(c, alpha):
    p = 1 - pow(1-c,1/n) #the probability of being a slot leader for a validator
    m = int((1-alpha) * n)
    b = binom(m,p)
    sum = 0
    for k in range(1,m+1):
        sum = sum + b.pmf(k)
    return pow(1-c, alpha) * sum
    
#finds the probability that at least one malicious validator is selected    
def prm(c, alpha):
    return c - prH(c,alpha)

def prOneH(c,alpha):
    p = 1 - pow(1-c,1/n) #the probability of being a slot leader for a validator
    return roundtwo(alpha * n * p * pow(1-p, n-1))

def aura_prob_honest_block(stake):
    return 1-stake

def babe_prob_honest_block(stake):
    return prH(c, 1-stake)

def babe_plus_aura_prob_only_honest_collator(stake):
    probability_babe_only_honest_collator = babe_prob_honest_block(stake)
    probability_aura_only_honest_collator = aura_prob_honest_block(stake)
    return probability_babe_only_honest_collator + (1-c)*probability_aura_only_honest_collator

algo_switcher = {
        'aura': aura_prob_honest_block,
        'babe': babe_prob_honest_block,
        'babe+aura': babe_plus_aura_prob_only_honest_collator
}

parser = argparse.ArgumentParser(description='Calculate stake probability')
parser.add_argument('--active_collators', dest='active_collators', type=int, default=max,
                    help='number of active collators')
parser.add_argument('--stake_steps', dest='stake_steps', default=max, type=float,
                    help='stake steps by attacker')
parser.add_argument('--stall_steps', dest='stall_steps', type=float,
                    help='stall steps to consider') 
parser.add_argument('--difficulty_parameter', dest='difficulty_parameter', type=float,
                    help='difficulty parameter to consider') 
parser.add_argument('--number_of_trials', dest='number_of_trials', type=float,
                    help='number of trials to run') 
parser.add_argument('--algorithm', dest='algorithm', type=str,
                    help='aura|babe|babe+aura|') 

args = parser.parse_args()
n = args.active_collators
steps = np.arange(args.stake_steps, 1, args.stake_steps)
stall_steps = np.arange(args.stall_steps, 1, args.stall_steps)
number_of_trials = args.number_of_trials
results = np.zeros((len(stall_steps), len(steps)))
c = args.difficulty_parameter

for i, stall_percentage in enumerate(stall_steps):
    for j, stake in enumerate(steps):
        single_stall_probability =  algo_switcher.get(args.algorithm, lambda: "Invalid algorithm")(stake)
        results[i][j] = binom_test(number_of_trials*stall_percentage, number_of_trials, single_stall_probability, 'greater')

labels = []
for stall in stall_steps:
    labels.append(str("%.2f" % (stall*100)) +  ('% of blocks proposed by honest actors'))

for y_arr, label in zip(results, labels):
    plt.plot(steps, y_arr, label=label)

plt.xlabel("Attacker controlled stake in active collator set")
plt.ylabel("Probability")
plt.legend()
plt.show()

