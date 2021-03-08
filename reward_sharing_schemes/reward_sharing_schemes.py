from scipy.stats import pareto
import matplotlib.pyplot as plt
import numpy as np
import copy
from random import randint
class Pool(object):
    stake = 0
    leader_stake = 0
    cost = 0
    margin = 0
    rank = 0
    # The class "constructor" - It's actually an initializer 
    def __init__(self, stake, leader_stake, cost, margin):
        self.stake = stake
        self.leader_stake = leader_stake
        self.cost = cost
        self.margin = margin
        self.rank = 0

class Staker(object):
    stake = 0
    cost = 0
    # The class "constructor" - It's actually an initializer 
    def __init__(self, stake, cost):
        self.stake = stake
        self.cost = cost

beta = 1/10
total_reward = 10
sybil_parameter = 0.02


def reward_function(pool_stake, leader_stake):
  sigma_prime = min(pool_stake, beta)
  lambda_prime = min(leader_stake, beta)
  right_hand_side = sigma_prime  + (lambda_prime*sybil_parameter*(sigma_prime-(lambda_prime*(1-(sigma_prime/beta))))/beta)
  left_hand_side = total_reward/(1+ sybil_parameter)
  return right_hand_side*left_hand_side

def leader_utility(pool):
  if pool.leader_stake == 0:
    return 0
  if reward_function(myopic_stake(pool), pool.leader_stake) < pool.cost:
    return reward_function(beta, pool.leader_stake) - pool.cost
  else:
    return (reward_function(myopic_stake(pool), pool.leader_stake) - pool.cost)*(pool.margin + ((1-pool.margin)*pool.leader_stake/myopic_stake(pool)))


def myopic_stake(pool):
  if pool.rank < 1/beta:
    return max(beta, pool.stake)
  else:
    return pool.leader_stake

def delegated_utility(pool, stake):
  if pool.stake == 0:
    return 0
  elif pool.rank >= 1/beta:
    return (1-pool.margin)*(reward_function(pool.leader_stake+stake, pool.leader_stake)-pool.cost)*(stake/(stake + pool.leader_stake))
  else:
    return (1-pool.margin)*max((reward_function(beta, pool.leader_stake)-pool.cost), 0)*(stake/myopic_stake(pool))

def compute_ranking(desirabilities):
  temp = copy.deepcopy(desirabilities)
  desirabilities.sort(reverse=True)
  ranking = []
  for item in temp:
    
    index = desirabilities.index(item)
    while index in ranking:
      index = desirabilities.index(item, index+1)  
    
    ranking.append(index)
  return ranking  
  
  np.argsort(desirabilities)
  
def compute_desirability(pool):
  profitability = reward_function(beta, pool.leader_stake) - pool.cost
  if profitability > 0:
    return ((1-pool.margin)*profitability)
  else:
    return 0

def calculate_utility(pools, delegations, player):
  if pools.get(player):
    return leader_utility(pools[player])
  else:
    new_utility = 0
    my_delegation = delegations[player]
    for delegation, stake in my_delegation.items():
        if temp_pools.get(delegation):
          new_utility += delegated_utility(temp_pools[delegation], stake)
    return new_utility

def open_pool_strategy(pools, player, past_utility, stake, cost):
  desirabilities = []
  temp_pools = copy.deepcopy(pools)
  
  first_try = Pool(beta, stake, cost, 1)
  temp_pools[player] = first_try

  desirabilities = []
  for pl, pool in temp_pools.items():
      desirabilities.append((compute_desirability(pool), reward_function(beta, pool.leader_stake) - pool.cost))
  ranking = compute_ranking(copy.deepcopy(desirabilities))
  keys = [*temp_pools]

  first_uti = leader_utility(first_try)
  if ranking[keys.index(player)] < 1/beta:
    if pools.get(player):
      return (1, first_uti)
    else:
      if (first_uti > past_utility + (10**-8) or len(pools) < 1/beta):
        return (1, first_uti)

  q = stake/max(stake, beta)
  reward = reward_function(beta, stake)
  if q == 1:
    m_prime_minimum = 1
  else:
    m_prime_minimum = (past_utility - ((reward-cost)*q))/((reward-cost)*(1-q))
  if m_prime_minimum < 0:
    m_prime_minimum = 0
  
  second_try = Pool(beta, stake, cost, m_prime_minimum)
  temp_pools[player] = second_try
  second_uti = leader_utility(second_try)
  
  desirabilities = []
  for pl, pool in temp_pools.items():
      desirabilities.append((compute_desirability(pool), reward_function(beta, pool.leader_stake) - pool.cost))
  ranking = compute_ranking(desirabilities)
  if ranking[keys.index(player)] < 1/beta:
    if pools.get(player):
      return (m_prime_minimum, second_uti)
    else:
      if second_uti > past_utility + (10**-8):
        return (m_prime_minimum, second_uti)
  
  return (0, 0)

def delegate_strategy(current_delegations, pools, past_utility, stake, cost):

  delegations = {}
  temp_pools = copy.deepcopy(pools)
  for delegation , delegated_stake in current_delegations.items():
    if temp_pools.get(delegation):
      temp_pools[delegation].stake -= delegated_stake

  total_delegated = 0
  resolution = 10**(-4)
  while total_delegated < stake:
    to_stake = resolution if stake-total_delegated > resolution else stake-total_delegated
    desirabilities = []
    for player, pool in temp_pools.items():
      desirabilities.append((compute_desirability(pool), reward_function(beta, pool.leader_stake) - pool.cost))
    ranking = compute_ranking(desirabilities)
    keys = [*pools]
    for player, pool in temp_pools.items():
      pool.rank = ranking[keys.index(player)]

    utilities = []
    for player, pool in temp_pools.items():
      utilities.append(delegated_utility(pool, resolution))

    highest_utility = utilities.index(max(utilities))


    if delegations.get(keys[highest_utility]):
      delegations[keys[highest_utility]] += to_stake
    else:
      delegations[keys[highest_utility]] = to_stake
    keys = [*temp_pools]

    temp_pools[keys[highest_utility]].stake += to_stake
    total_delegated += to_stake
  
  new_utility = 0

 
  for delegation, delegated_stake in delegations.items():
    new_utility += delegated_utility(temp_pools[delegation], delegated_stake)

  if new_utility > past_utility+10**(-8):
    return (delegations, new_utility, temp_pools, True)
  else:
    return (current_delegations, past_utility, pools, False)
    
total_pools = 10
total_players = 100
num_iterations = 20000
c_min = 0.0001
c_max = 0.0002
stake = pareto.rvs(2, size=total_players)
divide_factor = max(stake)*(total_pools)
#truncated_pareto = [x / divide_factor for x in stake]
truncated_pareto = [x / sum(stake) for x in stake]
count, bins, _ = plt.hist(truncated_pareto, 100, density=True)
plt.plot(bins,linewidth=2, color='r')
plt.show()
results = []
stop=False
results_len_pools = []
delegations = []
for i in range(0, total_players):
  delegations.append({})

costs = np.random.uniform(c_min, c_max, total_players)
utilities = [0] * total_players
print(truncated_pareto)
pools = {}
runs_without_changes = 500

for i in range(0, num_iterations):
  if runs_without_changes > 1000:
    break
  continue_in_iteration = True
  while continue_in_iteration:
    print("RUNS WITHOUT CHANGES %", runs_without_changes)
    if runs_without_changes > 500:
      break
    player = randint(0, len(stake)-1)
    before_pareto = truncated_pareto[player]
    (m, new_utility) = open_pool_strategy(pools,player, utilities[player], truncated_pareto[player], costs[player])
    if len(delegations[player]) != 0:
      (temp_delegations, temp_uti, temp_pools, updated) = delegate_strategy(delegations[player], pools, utilities[player], truncated_pareto[player], costs[player])
    else:
      temp_uti = 0
    if new_utility > utilities[player] + 10**-8 and new_utility> temp_uti:
       
      before_stake = sum(pools[pool].stake for pool in pools)
      before_delegated = sum(sum(delegation.values())  for delegation in delegations)
      before_pools = copy.deepcopy(pools)
          
      if pools.get(player):
        pools[player] = Pool(pools[player].stake, truncated_pareto[player], costs[player], m)
      else:
        print("OPENING %", player)
        pools[player] = Pool(truncated_pareto[player], truncated_pareto[player], costs[player], m)
      
        for key in delegations[player].keys():
          pools[key].stake -= delegations[player][key]
        delegations[player] = {}
        runs_without_changes = 0

      utilities[player] = new_utility
      continue_in_iteration = False
    else:
      print("DELEGATING %", player)
      before_stake = sum(pools[pool].stake for pool in pools)
      before_delegated = sum(sum(delegation.values())  for delegation in delegations)
      before_pools = copy.deepcopy(pools)
      before_delegations = copy.deepcopy(delegations)

      # Fix: remaining iteration does not get added
      (delegations[player], utilities[player], pools, updated) = delegate_strategy(before_delegations[player], before_pools, utilities[player], truncated_pareto[player], costs[player])
      
      if updated:
        if pools.get(player):
          del pools[player]
          for item in delegations:
            if player in item.keys():
              del item[player]
        
        continue_in_iteration = False
      runs_without_changes += 1  
      new_stake = sum(pools[pool].stake for pool in pools)
      new_delegated = sum(sum(delegation.values())  for delegation in delegations)
  desirabilities = []
  temp_pools = copy.deepcopy(pools)
  for index, pool in temp_pools.items():
      desirabilities.append((compute_desirability(pool), reward_function(beta, pool.leader_stake) - pool.cost))
  
  ranking = compute_ranking(desirabilities)
  keys = [*pools]
  for index, pool in pools.items():
    pool.rank = ranking[keys.index(index)]
  for index in range(0, total_players):
    utilities[index] = calculate_utility(pools, delegations, index)

  results.append(sum(pools[pool].stake for pool in pools))
  results_len_pools.append(len(pools))

#sum(sum(delegation.values())  for delegation in delegations)


plt.plot(range(0,num_iterations), results, color='r')



