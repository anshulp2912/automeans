# automeans
Python Library for automating the scikit-learn K-Means Clustering Algorithm by optimising the selection of number of clusters.

## Introduction
Kmeans algorithm is an iterative algorithm that tries to partition the dataset into K pre-defined distinct non-overlapping subgroups (clusters) where each data point belongs to only one group. It tries to make the intra-cluster data points as similar as possible while also keeping the clusters as different (far) as possible. It assigns data points to a cluster such that the sum of the squared distance between the data points and the cluster’s centroid (arithmetic mean of all the data points that belong to that cluster) is at the minimum. The less variation we have within clusters, the more homogeneous (similar) the data points are within the same cluster.

Problem : As the name suggest, Kmeans algorithm depends upon 'K' which often times is not known by the user at the time of fitting the data.
Solution : This library helps automate the selection process for the optimal number of cluster on a given data, along with an optimal fitted model.

## Features
- All the main features are adopted from Scikit-Learn Kmeans Algorithm
- Getting the optimally fitted kmeans model
- Getting the optimal number of cluster

## Installation
```Python
pip install automeans
```
## Usage
Importing the model
```python
from automeans.cluster import ameans
```
There are 3 metrics to choose from ['standard','kneed','silhouette']

- standard
```python
# Initialize the model
AM = ameans(max_clusters = 5, metrics = 'standard')
# Fit on data 'X'
model, cluster = AM.fit(X)
```
- kneed
```python
# Initialize the model
AM = ameans(max_clusters = 5, metrics = 'kneed')
# Fit on data 'X'
model, cluster = AM.fit(X)
```
- silhouette
```python
# Initialize the model
AM = ameans(max_clusters = 5, metrics = 'silhouette')
# Fit on data 'X'
model, cluster = AM.fit(X)
```
## Parameters
For initializing the model
max_clusters : The number of maximum seeds to choose
metrics : {"standard", "kneed", "silhouette"}, default="standard"
        Metric to choose the best number of cluster
All other parameters are same as used in [sklearn Kmeans algorithm](https://scikit-learn.org/dev/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)

## Example
```python
import numpy as np
X = np.array([[1, 2], [2, 5], [3, 6], [8, 7], [8, 8], [7, 3]])

from automeans.cluster import ameans
# Initialize the model
AM = ameans(max_clusters = 5, metrics = 'silhouette')
# Fit on data 'X'
model, cluster = AM.fit(X)
# Predict the cluster on data 'X'
predictions = model.predict(X)
```
## Acknowledgement
- [Standard Method](https://www.linkedin.com/pulse/finding-optimal-number-clusters-k-means-through-elbow-asanka-perera/)
- [Kneed Method](https://github.com/arvkevi/kneed)
- [Silhouette Method](https://stackoverflow.com/questions/54936518/how-do-i-automate-the-number-of-clusters)
- [Scikit-Learn](https://scikit-learn.org/dev/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)

## Licencse

MIT License

Copyright (c) 2020 Anshul Patel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
