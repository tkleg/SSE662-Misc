#All equations pulled from https://research.ijcaonline.org/volume48/number18/pxc3880534.pdf

import numpy as np
from scipy.optimize import brentq
from matplotlib import pyplot as plt
import pandas as pd

def getRandIntervals(N, phi, size, rng=None):
    #Set seed for reproducibility
    #Seed picked after it was randomly made and produced good results
    #np.random.seed(71139001)
    
    if rng is None:
        rng = np.random.default_rng()

    intervals = []
    for i in range(1, size + 1):
        lambda_i = generateFailureRate(N, phi, i)
        t_i = rng.exponential(1 / lambda_i)
        intervals.append(t_i)
    return np.array(intervals)

def generateFailureRate(N, phi, intervalNum):
    return phi * (N - (intervalNum - 1))

def estimateParameters(intervals):
    n = len(intervals)
    #Solve for N^ using equation (6)
    def eq6Left(NEst):
        return sum( 1 / (NEst - (i - 1)) for i in range(1, n + 1) )
    def eq6Right(NEst):
        sumIntervalsInv = 1 / np.sum(intervals)
        sumWeightedIntervals = sum( (i - 1) * intervals[i - 1] for i in range(1, n + 1) )
        return n / ( NEst - sumWeightedIntervals * sumIntervalsInv )
    def eq6(NEst):
        return eq6Left(NEst) - eq6Right(NEst)
    
    #Ensure different signs for root finding call
    left = n + 1e-5
    right = n + 1e-4
    while eq6(left) * eq6(right) > 0:
        right += 1e-2

    #Find the root
    NEst = brentq(eq6, left, right)

    #Solve for phi^ using equation (5)
    phiEstBottom = NEst * np.sum(intervals) - sum( (i - 1) * intervals[i - 1] for i in range(1, n + 1) )
    phiEst = n / phiEstBottom

    return NEst, phiEst


#Handle random seed
seed = np.random.randint(0, 2**31)
#seed = 1801421561
rng = np.random.default_rng(seed)

#Create random time intervals
intervals = getRandIntervals(100, 0.1, 75, rng)


NReal = 100 #Number of failures
phiReal = 0.1 #Failure rate contributed by each fault

#Estimate parameters
NEst, phiEst = estimateParameters(intervals)

estiamtedFailureRates = np.array([generateFailureRate(NEst, phiEst, i) for i in range(1, len(intervals) + 1)])
actualFailureRates = np.array([generateFailureRate(NReal, phiReal, i) for i in range(1, len(intervals) + 1)])

#Plot the failure rates
plt.plot(estiamtedFailureRates, label='Estimated Failure Rate')
plt.plot(actualFailureRates, label='Actual Failure Rate')
plt.xlabel('Failure Number')
plt.ylabel('Failure Rate')
plt.grid()
plt.legend()
plt.title("Estimated vs Actual Failure Rates")

plt.savefig('failure_rates.png')


#Round data for better readability
intervals = np.round(intervals, 4)
estiamtedFailureRates = np.round(estiamtedFailureRates, 4)
actualFailureRates = np.round(actualFailureRates, 4)
percentDifference = np.abs(np.round(100 * (estiamtedFailureRates - actualFailureRates) / actualFailureRates, 2))

#Create a DataFrame and write to a CSV file
pd.DataFrame({
    'Failure Number': np.arange(1, len(intervals) + 1),
    'Time Interval': intervals,
    'Estimated Failure Rate': estiamtedFailureRates,
    'Actual Failure Rate': actualFailureRates,
    'Percent Difference': percentDifference
}).to_csv('failure_rates.csv', index=False)

#Save nontable data to a text file
with open('results.txt', 'w') as f:
    f.write(f"Estimated N: {NEst:.2f}, Estimated phi: {phiEst:.4f}\n")
    f.write(f"Actual N: {NReal}, Actual phi: {phiReal:.4f}\n")
    f.write(f"Random seed used: {seed}\n")