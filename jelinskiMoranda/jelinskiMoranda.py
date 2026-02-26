#All equations pulled from https://research.ijcaonline.org/volume48/number18/pxc3880534.pdf

import numpy as np
from scipy.optimize import brentq
from matplotlib import pyplot as plt
import pandas as pd
from math import exp

def getRandIntervals(N, phi, size, rng=None):
    #Set seed for reproducibility
    #Seed picked after it was randomly made and produced good results
    #np.random.seed(71139001)
    
    if rng is None:
        rng = np.random.default_rng()

    intervals = []
    for i in range(1, size + 1):
        lambda_i = calcFailureRate(N, phi, i)
        t_i = rng.exponential(1 / lambda_i)
        intervals.append(t_i)
    return np.array(intervals)

def calcFailureRate(N, phi, intervalNum):
    return phi * (N - (intervalNum - 1))

def calcFailureDensity(N, phi, intervalNum, t):
    return calcFailureRate(N, phi, intervalNum) * exp( -phi * ( N - (intervalNum - 1) ) * t )

def calcFailureDistribution(N, phi, intervalNum, t):
    return 1 - calcReliability(N, phi, intervalNum, t)

def calcReliability(N, phi, intervalNum, t):
    return exp( -phi * ( N - (intervalNum - 1) ) * t )

def calcMeanTimeToFailure(N, phi, intervalNum):
    return 1 / calcFailureRate(N, phi, intervalNum)

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
#seed = np.random.randint(0, 2**31)
seed = 1749434685
rng = np.random.default_rng(seed)

NReal = 250 #Number of failures - must match N used in getRandIntervals
phiReal = 0.1 #Failure rate contributed by each fault

#Create random time intervals
intervals = getRandIntervals(NReal, phiReal, NReal-1, rng)
print("Done making intervals", intervals)


#Estimate parameters
NEst, phiEst = estimateParameters(intervals)
print("Done estimating parameters")

estimatedFailureRates = np.array([calcFailureRate(NEst, phiEst, i) for i in range(1, len(intervals) + 1)])
actualFailureRates = np.array([calcFailureRate(NReal, phiReal, i) for i in range(1, len(intervals) + 1)])
print("Done calculating failure rates")

estimatedFailureDensities = np.array([calcFailureDensity(NEst, phiEst, i, intervals[i - 1]) for i in range(1, len(intervals) + 1)])
actualFailureDensities = np.array([calcFailureDensity(NReal, phiReal, i, intervals[i - 1]) for i in range(1, len(intervals) + 1)])
print("Done calculating failure densities")

estimatedFailureDistributions = np.array([calcFailureDistribution(NEst, phiEst, i, intervals[i - 1]) for i in range(1, len(intervals) + 1)])
actualFailureDistributions = np.array([calcFailureDistribution(NReal, phiReal, i, intervals[i - 1]) for i in range(1, len(intervals) + 1)])
print("Done calculating failure distributions")

estimatedReliabilities = np.array([calcReliability(NEst, phiEst, i, intervals[i - 1]) for i in range(1, len(intervals) + 1)])
actualReliabilities = np.array([calcReliability(NReal, phiReal, i, intervals[i - 1]) for i in range(1, len(intervals) + 1)])
print("Done calculating reliabilities")

estimatedMeanTimesToFailure = np.array([calcMeanTimeToFailure(NEst, phiEst, i) for i in range(1, len(intervals) + 1)])
actualMeanTimesToFailure = np.array([calcMeanTimeToFailure(NReal, phiReal, i) for i in range(1, len(intervals) + 1)])
print("Done calculating mean times to failure")

#Plot the failure rates
plt.plot(estimatedFailureRates, label='Estimated Failure Rate')
plt.plot(actualFailureRates, label='Actual Failure Rate')
plt.xlabel('Failure Number')
plt.ylabel('Failure Rate')
plt.grid()
plt.legend()
plt.title("Estimated vs Actual Failure Rates")
plt.savefig('jelinskiMoranda/failure_rates.png')

#Plot the failure densities
plt.figure()
plt.plot(estimatedFailureDensities, label='Estimated Failure Density')
plt.plot(actualFailureDensities, label='Actual Failure Density')
plt.xlabel('Failure Number')
plt.ylabel('Failure Density')
plt.grid()
plt.legend()
plt.title("Estimated vs Actual Failure Densities")
plt.savefig('jelinskiMoranda/failure_densities.png')

#Plot the failure distributions
plt.figure()
plt.plot(estimatedFailureDistributions, label='Estimated Failure Distribution')
plt.plot(actualFailureDistributions, label='Actual Failure Distribution')
plt.xlabel('Failure Number')
plt.ylabel('Failure Distribution')
plt.grid()
plt.legend()
plt.title("Estimated vs Actual Failure Distributions")
plt.savefig('jelinskiMoranda/failure_distributions.png')

#Plot the reliabilities
plt.figure()
plt.plot(estimatedReliabilities, label='Estimated Reliability')
plt.plot(actualReliabilities, label='Actual Reliability')
plt.xlabel('Failure Number')
plt.ylabel('Reliability')
plt.grid()
plt.legend()
plt.title("Estimated vs Actual Reliabilities")
plt.savefig('jelinskiMoranda/reliabilities.png')

#Plot the mean times to failure
plt.figure()
plt.plot(estimatedMeanTimesToFailure, label='Estimated Mean Time to Failure')
plt.plot(actualMeanTimesToFailure, label='Actual Mean Time to Failure')
plt.xlabel('Failure Number')
plt.ylabel('Mean Time to Failure')
plt.grid()
plt.legend()
plt.title("Estimated vs Actual Mean Times to Failure")
plt.savefig('jelinskiMoranda/mean_times_to_failure.png')

#Round data for better readability
intervals = np.round(intervals, 4)
estimatedFailureRates = np.round(estimatedFailureRates, 4)
actualFailureRates = np.round(actualFailureRates, 4)
failureRatePercentDifference = np.abs(np.round(100 * (estimatedFailureRates - actualFailureRates) / actualFailureRates, 2))
estimatedFailureDensities = np.round(estimatedFailureDensities, 4)
actualFailureDensities = np.round(actualFailureDensities, 4)
failureDensityPercentDifference = np.abs(np.round(100 * (estimatedFailureDensities - actualFailureDensities) / actualFailureDensities, 2))
estimatedFailureDistributions = np.round(estimatedFailureDistributions, 4)
actualFailureDistributions = np.round(actualFailureDistributions, 4)
failureDistributionPercentDifference = np.abs(np.round(100 * (estimatedFailureDistributions - actualFailureDistributions) / actualFailureDistributions, 2))
estimatedReliabilities = np.round(estimatedReliabilities, 4)
actualReliabilities = np.round(actualReliabilities, 4)
reliabilityPercentDifference = np.abs(np.round(100 * (estimatedReliabilities - actualReliabilities) / actualReliabilities, 2))
estimatedMeanTimesToFailure = np.round(estimatedMeanTimesToFailure, 4)
actualMeanTimesToFailure = np.round(actualMeanTimesToFailure, 4)
meanTimeToFailurePercentDifference = np.abs(np.round(100 * (estimatedMeanTimesToFailure - actualMeanTimesToFailure) / actualMeanTimesToFailure, 2))

#Create a DataFrame and write to a CSV file
pd.DataFrame({
    'Failure Number': np.arange(1, len(intervals) + 1),
    'Time Interval': intervals,
    'Estimated Failure Rate': estimatedFailureRates,
    'Actual Failure Rate': actualFailureRates,
    'Failure Rate Percent Difference': failureRatePercentDifference,
    'Estimated Failure Density': estimatedFailureDensities,
    'Actual Failure Density': actualFailureDensities,
    'Failure Density Percent Difference': failureDensityPercentDifference,
    'Estimated Failure Distribution': estimatedFailureDistributions,
    'Actual Failure Distribution': actualFailureDistributions,
    'Failure Distribution Percent Difference': failureDistributionPercentDifference,
    'Estimated Reliability': estimatedReliabilities,
    'Actual Reliability': actualReliabilities,
    'Reliability Percent Difference': reliabilityPercentDifference,
    'Estimated Mean Time to Failure': estimatedMeanTimesToFailure,
    'Actual Mean Time to Failure': actualMeanTimesToFailure,
    'Mean Time to Failure Percent Difference': meanTimeToFailurePercentDifference
}).to_csv('jelinskiMoranda/data.csv', index=False)

#Save nontable data to a text file
with open('jelinskiMoranda/results.txt', 'w') as f:
    f.write(f"Estimated N: {NEst:.2f}, Estimated phi: {phiEst:.4f}\n")
    f.write(f"Actual N: {NReal}, Actual phi: {phiReal:.4f}\n")
    f.write(f"Random seed used: {seed}\n")