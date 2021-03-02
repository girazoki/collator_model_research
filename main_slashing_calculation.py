import argparse
import numpy as np
import matplotlib.pyplot as plt
import math

from scipy.stats import binom, binom_test
import scipy.special

from eligibility import vrf_eligibility, babe_probability, vrf_eligibility_exact

n = 0 #number of collators
alpha = 0 #honest collators
c = 0 #difficulty parameter

def single_honest_babe():
    babe_prob = babe_probability(n, c)
    single_val = vrf_eligibility(1, babe_prob, 1, "greater")
    no_other = vrf_eligibility(n-1, babe_prob, 0, "less")
    return single_val*no_other

def single_honest_babe_plus_aura_without_prio():

    babe_prob = babe_probability(n, c)
    single_hon_babe = single_honest_babe()
    no_one_babe = vrf_eligibility(n, babe_prob, 0, "less")
    return single_hon_babe + ((no_one_babe/n))

def chernof_bound_calculator(single_probability, num_trials, target):
    average = single_probability*num_trials
    return math.exp(-1*((expected-target)**2)/(2*expected))

algo_switcher = {
    'babe': single_honest_babe,
    'babe+aura': single_honest_babe_plus_aura_without_prio
}

parser = argparse.ArgumentParser(description='Calculate stake probability')
parser.add_argument('--active_collators_step', dest='active_collators_step', type=int, default=max,
                    help='number of active collators')
parser.add_argument('--difficulty_parameter', dest='difficulty_parameter', type=float,
                    help='difficulty parameter to consider') 
parser.add_argument('--number_of_trials', dest='number_of_trials', type=int,
                    help='number of trials to run')
parser.add_argument('--algorithm', dest='algorithm', type=str,
                    help='aura|babe|babe+aura|') 

args = parser.parse_args()
number_of_trials = args.number_of_trials
c = args.difficulty_parameter
collator_steps = np.arange(args.active_collators_step, 100, args.active_collators_step,)

results = []

for i, active_collators in enumerate(collator_steps):
    n = active_collators
    single_prob = algo_switcher.get(args.algorithm, lambda: "Invalid algorithm")()
    expected = single_prob*number_of_trials
    res = np.zeros(math.floor(expected))
    for j in range(1, math.floor(expected)+1):
        res[j-1] = chernof_bound_calculator(single_prob, number_of_trials, j)
    plt.plot(range(1, math.floor(expected)+1), res, label=str(active_collators)+  ('number of collators'))


plt.xlabel("Target number of blocks")
plt.ylabel("Probability bound by chernof")
plt.legend()
plt.show()

