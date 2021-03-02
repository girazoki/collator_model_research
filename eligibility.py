from scipy import stats


def vrf_eligibility(trials, probability, positive_outcomes, alternative):
    return stats.binom_test(positive_outcomes, trials, probability, alternative)

def babe_probability(num_validators, security_parameter):
    return 1-(1-security_parameter)**(1/num_validators)

def vrf_eligibility_exact(trials, probability, positive_outcomes):
    return (vrf_eligibility(trials, probability, positive_outcomes+1, 'less') - vrf_eligibility(trials, probability, positive_outcomes, 'less'))