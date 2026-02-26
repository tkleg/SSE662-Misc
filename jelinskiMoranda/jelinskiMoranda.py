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

#Reference: S.Mahapatra & Roy, 2012, pp. 39
def calcFailureRate(N, phi, intervalNum):
    return phi * (N - (intervalNum - 1))

#Reference: S.Mahapatra & Roy, 2012, pp. 39
def calcFailureDensity(N, phi, intervalNum, t):
    return calcFailureRate(N, phi, intervalNum) * exp( -phi * ( N - (intervalNum - 1) ) * t )

#Reference: S.Mahapatra & Roy, 2012, pp. 39
def calcFailureDistribution(N, phi, intervalNum, t):
    return 1 - calcReliability(N, phi, intervalNum, t)

#Reference: S.Mahapatra & Roy, 2012, pp. 39
def calcReliability(N, phi, intervalNum, t):
    return exp( -phi * ( N - (intervalNum - 1) ) * t )

#Reference: S.Mahapatra & Roy, 2012, pp. 39
def calcMeanTimeToFailure(N, phi, intervalNum):
    return 1 / calcFailureRate(N, phi, intervalNum)

#Reference: S.Mahapatra & Roy, 2012, pp. 39
#Reference is for the basic equations, not the method of solving or root finding
def estimateParameters(intervals):
    n = len(intervals)
    
    def eq6Left(NEst):
        return sum( 1 / (NEst - (i - 1)) for i in range(1, n + 1) )

    def eq6Right(NEst):
        sumIntervalsInv = 1 / np.sum(intervals)
        sumWeightedIntervals = sum( (i - 1) * intervals[i - 1] for i in range(1, n + 1) )
        return n / ( NEst - sumWeightedIntervals * sumIntervalsInv )
    
    #Moves both sides to the left to solve with root finding, as the dependent variable is in a summutation, making analytical solving difficult
    def eq6(NEst):
        return eq6Left(NEst) - eq6Right(NEst)
    
    #Ensure different signs for root finding call
    print("Finding root for N...")
    left = n + 1e-5
    right = left + 0.1
    while eq6(left) * eq6(right) > 0:
        right += 0.1
    print(f"Root found between {left:.5f} and {right:.5f}")

    #Find the root
    NEst = brentq(eq6, left, right)

    #Solve for phi^ using equation (5)
    phiEstBottom = NEst * np.sum(intervals) - sum( (i - 1) * intervals[i - 1] for i in range(1, n + 1) )
    phiEst = n / phiEstBottom

    return NEst, phiEst


#Handle random seed
#seed = np.random.randint(0, 2**31)
#seed changed from random to hardcoded once I found one that produced good results, to ensure reproducibility
seed = 1749434685
rng = np.random.default_rng(seed)

NReal = 250 #Number of failures - must match N used in getRandIntervals
phiReal = 0.1 #Failure rate contributed by each fault

#Create random time intervals
intervals = getRandIntervals(NReal, phiReal, NReal-1, rng)
print("Done making intervals")

#Estimate parameters
NEst, phiEst = estimateParameters(intervals)
print("Done estimating parameters")

graphs = {
    'failureRate': {
        "xlabel": 'Failure Number',
        "ylabel": 'Failure Rate',
        "title": "Estimated vs Actual Failure Rates",
        "estimated": np.array([calcFailureRate(NEst, phiEst, i) for i in range(1, len(intervals) + 1)]),
        "actual": np.array([calcFailureRate(NReal, phiReal, i) for i in range(1, len(intervals) + 1)]),
    },
    'failureDensity': {
        "xlabel": 'Failure Number',
        "ylabel": 'Failure Density',
        "title": "Estimated vs Actual Failure Densities",
        "estimated": np.array([calcFailureDensity(NEst, phiEst, i, intervals[i - 1]) for i in range(1, len(intervals) + 1)]),
        "actual": np.array([calcFailureDensity(NReal, phiReal, i, intervals[i - 1]) for i in range(1, len(intervals) + 1)])
    },
    'failureDistribution': {
        "xlabel": 'Failure Number',
        "ylabel": 'Failure Distribution',
        "title": "Estimated vs Actual Failure Distributions",
        "estimated": np.array([calcFailureDistribution(NEst, phiEst, i, 0.1) for i in range(1, len(intervals) + 1)]),
        "actual": np.array([calcFailureDistribution(NReal, phiReal, i, 0.1) for i in range(1, len(intervals) + 1)])
    },
    'reliability': {
        "xlabel": 'Failure Number',
        "ylabel": 'Reliability',
        "title": "Estimated vs Actual Reliabilities",
        "estimated": np.array([calcReliability(NEst, phiEst, i, 0.1) for i in range(1, len(intervals) + 1)]),
        "actual": np.array([calcReliability(NReal, phiReal, i, 0.1) for i in range(1, len(intervals) + 1)])
    },
    'meanTimeToFailure': {
        "xlabel": 'Failure Number',
        "ylabel": 'Mean Time to Failure',
        "title": "Estimated vs Actual Mean Times to Failure",
        "estimated": np.array([calcMeanTimeToFailure(NEst, phiEst, i) for i in range(1, len(intervals) + 1)]),
        "actual": np.array([calcMeanTimeToFailure(NReal, phiReal, i) for i in range(1, len(intervals) + 1)])
    }
}
#Calculate percent difference for each graph
for fileName, graph in graphs.items():
    graphs[fileName]['percentDifference'] = ((graph['estimated'] - graph['actual']) / graph['actual']) * 100

#Plot the data
for fileName, graph in graphs.items():
    plt.figure(dpi=750)
    plt.plot(graph['estimated'], label='Estimated', linewidth=0.75, alpha=0.8)
    plt.plot(graph['actual'], label='Actual', linewidth=0.75, alpha=0.8)
    plt.xlabel(graph['xlabel'])
    plt.ylabel(graph['ylabel'])
    plt.title(graph['title'])
    plt.legend()
    plt.grid()
    plt.subplots_adjust(bottom=0.15)  # Add space at bottom for caption
    if fileName in ['reliability', 'failureDistribution']:
        plt.figtext(0.5, 0.05, "Time interval is fixed to 0.1 seconds for better visualization", 
                   ha='center', va='center', fontsize=9, style='italic')
    else:
        plt.figtext(0.5, 0.05, "Time interval is generated", 
                   ha='center', va='center', fontsize=9, style='italic')
    plt.savefig(f'jelinskiMoranda/{fileName}.png')


#Round data for better readability in csv
intervals = np.round(intervals, 4)
for graph in graphs.values():
    graph['estimated'] = np.round(graph['estimated'], 4)
    graph['actual'] = np.round(graph['actual'], 4)

#Create a DataFrame and write to a CSV file
df_dict = {
    'Failure Number': np.arange(1, len(intervals) + 1),
    'Time Interval': intervals,
}
df_dict.update({f"{graph['ylabel']} Estimated": graph['estimated'] for graph in graphs.values()})
df_dict.update({f"{graph['ylabel']} Actual": graph['actual'] for graph in graphs.values()})
df_dict.update({f"{graph['ylabel']} Percent Difference": graph['percentDifference'] for graph in graphs.values()})

def sortColumns(colName):
    colRoots = ['Failure Number', 'Time Interval', 'Failure Rate', 'Failure Density', 'Failure Distribution', 'Reliability', 'Mean Time to Failure']
    colExtensions = ['Estimated', 'Actual', 'Percent Difference']
    for root in colRoots:
        if colName.startswith(root):
            for ext in colExtensions:
                if colName.endswith(ext):
                    return (colRoots.index(root), colExtensions.index(ext))
            return (colRoots.index(root), 0) #Put columns without extensions at the front

df_dict = dict(sorted(df_dict.items(), key=lambda item: sortColumns(item[0])))

pd.DataFrame(df_dict).to_csv('jelinskiMoranda/data.csv', index=False)


#Remove Outliers from percent difference data for better graph scaling
#Use IQR
for fileName, graph in graphs.items():
    percentDiff = graph['percentDifference']
    q1 = np.percentile(percentDiff, 25)
    q3 = np.percentile(percentDiff, 75)
    iqr = q3 - q1
    lowerBound = q1 - 1.5 * iqr
    upperBound = q3 + 1.5 * iqr
    graph['percentDifference'] = percentDiff[(percentDiff >= lowerBound) & (percentDiff <= upperBound)]

#Save nontable data to a text file
with open('jelinskiMoranda/results.txt', 'w') as f:
    f.write(f"Random seed used: {seed}\n")
    f.write(f"Estimated N: {NEst:.2f}, Actual N: {NReal:.4f}, Percent Difference: {((NEst - NReal) / NReal) * 100:.2f}%\n")
    f.write(f"Estimated phi: {phiEst:.4f}, Actual phi: {phiReal:.4f}, Percent Difference: {((phiEst - phiReal) / phiReal) * 100:.2f}%\n")
    for fileName, graph in graphs.items():
        f.write(f"{graph['title']} - Average Percent Difference: {np.mean(graph['percentDifference']):.2f}%\n")