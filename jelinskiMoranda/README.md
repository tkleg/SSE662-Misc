# Jelinski-Moranda Model Case Study


## Purpose
To show a basic Jelinsk-Moranda model running, with the accuracy of several estimated metrics measured, graphed, and analyzed.

## Values Measured
- Failure Rate
- Failure Density
- Failure Distribution
- Reliability
- Mean time to Failure
  
## Methodology
### Data Generation
I attempted to source data from the paper I got the equations from, but the results were poor. I opted for an exponential distribution instead since the time intervals should be exponentially distributed.
#### Steps
1. Randomly generate exponentially distributed data with a mean of the inverse of the failure rate.
2. Once acceptable data is generated, hardcode the random generator seed to allow reproduction of values.

### Program
1. Generate time intervals using the hardcoded seed.
2. Estimate the parameters *N* and *$\phi$*.
3. Calculate estimated and actual arrays for each failure.
   - One value per array for every failure.
4. Generate percent differences for every entry of each estimated and actual array.
5. Graph each estimated array against the respective actual array and save the graph.
6. Save all data to a CSV file.
7. Filter out outliers from the percent differences using the interquartile range.
   - I chose to do this because as the values for estimated and absolute become very small, the percent difference explode, swaying the mean percent difference calculated in the next step.
8. Write the following data to a text file.
    - The random seed used.
    - Estimated, actual, and percent differences between estimated and actual for the parameters *N* and *$\phi$*.
    - Average percent difference between every estimated and actual array.
## Source of equations
The equations were pulled from a paper found [HERE](https://research.ijcaonline.org/volume48/number18/pxc3880534.pdf)

