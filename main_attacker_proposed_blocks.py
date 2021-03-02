import argparse
import numpy as np
import matplotlib.pyplot as plt
import math

from scipy.stats import binom, binom_test
import scipy.special

from eligibility import vrf_eligibility_exact, vrf_eligibility, babe_probability

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

def aura_prob_attacker_block(stake):
    return stake

def babe_prob_attacker_block(stake):
    return prM(c, 1-stake)

def babe_prob_attacker_no_priorization(stake):
    babe_prob = babe_probability(n, c)
    prob = 0
    for i in range(1, int(stake*n)+1):
        for j in range(0, int((1-stake)*n)+1):
          honest_nodes = vrf_eligibility_exact((1-stake)*n, babe_prob, j)
          attacker_nodes = vrf_eligibility_exact((stake)*n, babe_prob, i)
          prob += (i/(j+i)) * honest_nodes*attacker_nodes
    return prob

def babe_plus_aura_prob_only_attacker_collator(stake):
    probability_babe_only_attacker_collator = babe_prob_attacker_block(stake)
    probability_aura_only_attacker_collator = aura_prob_attacker_block(stake)
    return probability_babe_only_attacker_collator + (1-c)*probability_aura_only_attacker_collator

def babe_plus_aura_prob_attacker_no_priorization(stake):
    babe_prob = babe_probability(n, c)
    prob = 0
    for i in range(1, int(stake*n)+1):
        for j in range(0, int((1-stake)*n)+1):
          honest_nodes = vrf_eligibility_exact((1-stake)*n, babe_prob, j)
          attacker_nodes = vrf_eligibility_exact((stake)*n, babe_prob, i)
          # Case Aura selects attacker
          prob += (i+1/(j+i+1)) * honest_nodes*attacker_nodes*stake
          # Case Aura selects honest
          prob += (i/(j+i+1)) * honest_nodes*attacker_nodes*(1-stake)
    return prob


def slashing(max_slashing_percentage, attacker_nodes, total_nodes):
    return max_slashing_percentage*min(3*(attacker_nodes-1)/n,1)

algo_switcher = {
        'aura': aura_prob_attacker_block,
        'babe': babe_prob_attacker_block,
        'babe_without_prio': babe_prob_attacker_no_priorization,
        'babe+aura': babe_plus_aura_prob_only_attacker_collator,
        'babe+aura_without_prio': babe_plus_aura_prob_attacker_no_priorization
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate stake probability')
    parser.add_argument('--active_collators', dest='active_collators', type=int, default=max,
                    help='number of active collators')
    parser.add_argument('--stake_steps', dest='stake_steps', default=max, type=float,
                    help='stake steps by attacker')
    parser.add_argument('--difficulty_parameter', dest='difficulty_parameter', type=float,
                    help='difficulty parameter to consider') 
    parser.add_argument('--algorithm', dest='algorithm', type=str,
                    help='aura|babe|babe+aura|')
    parser.add_argument('--num_eras', dest='num_eras', type=int, default=max,
                    help='number of eras to be considered')
    parser.add_argument('--max_slashing_percentage', dest='max_slashing_percentage', type=float, default=max,
                    help='max slashing percentage to be considered') 

    args = parser.parse_args()
    n = args.active_collators
    steps = np.arange(args.stake_steps, 1, args.stake_steps)
    results = np.zeros((args.num_eras, len(steps)))
    results_stake = np.zeros((args.num_eras, len(steps)))

    c = args.difficulty_parameter

    for i in range(0, args.num_eras):
        print("Computing era %d" % i)
        for j, stake in enumerate(steps):
            single_stall_probability = algo_switcher.get(args.algorithm, lambda: "Invalid algorithm")(stake)
            results[i][j] = single_stall_probability
            slashing_percentage = slashing(args.max_slashing_percentage, stake*n, n)
            steps[j] = steps[j]*(1-slashing_percentage)
            results_stake[i][j] = steps[j]

    labels = []
    for stake in np.arange(args.stake_steps, 1, args.stake_steps):
        labels.append(str("%.2f" % (stake*100)) +  ('% share with which the attacker started'))

    for y_arr, label in zip(results.transpose(), labels):
        plt.plot(range(0, args.num_eras), y_arr, label=label)

    plt.xlabel("eras")
    plt.ylabel("Expected number of blocks proposed by the attacker")
    plt.legend()
    plt.show()

    labels = []
    for stake in np.arange(args.stake_steps, 1, args.stake_steps):
        labels.append(str("%.2f" % (stake*100)) +  ('% share with which the attacker started'))

    for y_arr, label in zip(results_stake.transpose(), labels):
        plt.plot(range(0, args.num_eras), y_arr, label=label)

    plt.xlabel("eras")
    plt.ylabel("Attacker share")
    plt.legend()
    plt.show()




