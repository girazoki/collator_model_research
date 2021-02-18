import numpy as np
import random
import matplotlib.pyplot as plt
import copy 
import argparse

# nominees is a tuple consisting in (stake, [candidates])
def weighted_phragmen(number_of_seats, nominators, candidates):
  stake_at_candidates = np.zeros(len(candidates))
  candidate_scores = np.zeros(len(candidates))
  voter_load = np.zeros(len(nominators))
  winners = []
  for (stake, candidates) in nominators:
      for candidate in candidates:
          stake_at_candidates[candidate] += stake
  
  for candidate,stake in enumerate(stake_at_candidates):
      candidate_scores[candidate] = 1./stake

  updated_candidate_scores = np.copy(candidate_scores)
  for i in range(number_of_seats):
    winner = np.argmin(updated_candidate_scores)
    winners.append(winner)
    winner_scores = updated_candidate_scores[winner]
    updated_candidate_scores = np.copy(candidate_scores)
    for i in range(0, len(nominators)):
        if winner in nominators[i][1]:
            voter_load[i] = winner_scores
            nominators[i][1].remove(winner)
        
        for nominated in nominators[i][1]:
            updated_candidate_scores[nominated] += ((nominators[i][0]*voter_load[i])/stake_at_candidates[nominated])
    for winner in winners:
        updated_candidate_scores[winner] = float('inf')

  return winners


parser = argparse.ArgumentParser(description='Calculate weighted phragmen')
parser.add_argument('--active_collator_set', dest='active_collators', type=int, default=max,
                    help='number of active collators')
parser.add_argument('--attacker_stake_steps', dest='attacker_stake_steps', default=max, type=float,
                    help='stake steps by attacker')
parser.add_argument('--total_supply', dest='total_supply', type=int,
                    help='total supply to consider')
parser.add_argument('--honest_stake_steps', dest='honest_stake_steps', type=float,
                    help='honest_stake steps to consider')                    

args = parser.parse_args()
honest_stake_array = np.arange(args.honest_stake_steps, 0.5, args.honest_stake_steps)
attacker_stake_array = np.arange(args.attacker_stake_steps, 1, args.attacker_stake_steps)

results = np.zeros((len(honest_stake_array), len(attacker_stake_array)))
for i, honest_stake in enumerate(honest_stake_array):
  initial_distribution = np.random.normal(honest_stake*args.total_supply/args.active_collators, 0.1*honest_stake*args.total_supply/args.active_collators, args.active_collators)
  nominators = []
  for stake in initial_distribution:
     num_candidates_to_back = random.randint(1,args.active_collators)
     nominators.append(tuple((stake, random.sample(range(0, args.active_collators), num_candidates_to_back))))

  for j, attacker_stake in enumerate(attacker_stake_array):
   attacker_share = attacker_stake*(1-honest_stake)*args.total_supply
   attacker_distribution = np.full(args.active_collators, attacker_share/(args.active_collators))
   candidates = args.active_collators*2
   distribution = copy.deepcopy(nominators)
   for index,share in enumerate(attacker_distribution):
     distribution.append(tuple((share, list(range(args.active_collators, args.active_collators*2)))))

   winners = weighted_phragmen(args.active_collators, distribution, np.arange(0, candidates))
   results[i][j] = sum(z >= args.active_collators for z in winners)

labels = []
for i in range(0, len(honest_stake_array)):
    labels.append(("%.2f" % (honest_stake_array[i]*100)) +  ('% of stake destinated to security'))

for y_arr, label in zip(results, labels):
    plt.plot(attacker_stake_array, y_arr, label=label)

plt.xlabel("Stake controlled by the attacker")
plt.ylabel("Malicious number of collators inserted by attacker")
plt.legend()
plt.show()