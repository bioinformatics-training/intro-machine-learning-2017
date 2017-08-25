# Clustering

<!-- Matt -->

<!-- http://www.madgroup.path.cam.ac.uk/microarraysummary.shtml -->

## Introduction

## Types of cluster

<div class="figure" style="text-align: center">
<img src="09-clustering_files/figure-html/clusterTypes-1.png" alt="Example clusters. **A**, *blobs*; **B**, *aggregation* [@Gionis2007]; **C**, *noisy moons*; **D**, *noisy circles*; **E**, *D31* [@Veenman2002]; **F**, *no structure*." width="80%" />
<p class="caption">(\#fig:clusterTypes)Example clusters. **A**, *blobs*; **B**, *aggregation* [@Gionis2007]; **C**, *noisy moons*; **D**, *noisy circles*; **E**, *D31* [@Veenman2002]; **F**, *no structure*.</p>
</div>

## Distance metrics


**Minkowski distance:**
\begin{equation}
  distance\left(x,y,p\right)=\left(\sum_{i=1}^{n} abs(x_i-y_i)^p\right)^{1/p}
  (\#eq:minkowski)
\end{equation}

## K-means

Pseudocode

to illustrate range of different types of data that can be clustered - image segmentation

### Image segmentation



## Hierarchic methods




Table: (\#tab:distance-matrix)Example distance matrix

     A    B    C    D  
---  ---  ---  ---  ---
B    2                 
C    6    5            
D    10   10   5       
E    9    8    3    4  

### Linkage algorithms
Make one section
panel of three dendrograms
one table

Single linkage - nearest neighbours linkage
Complete linkage - furthest neighbours linkage
Average linkage - UPGMA (Unweighted Pair Group Method with Arithmetic Mean) 






Table: (\#tab:distance-merge)Merge distances for objects in the example distance matrix using three different linkage methods.

Groups          Single   Complete   Average 
--------------  -------  ---------  --------
A,B,C,D,E       0        0          0       
(A,B),C,D,E     2        2          2       
(A,B),(C,E),D   3        3          3       
(A,B)(C,D,E)    4        5          4.5     
(A,B,C,D,E)     5        10         8       

<div class="figure" style="text-align: center">
<img src="09-clustering_files/figure-html/linkageComparison-1.png" alt="Dendrograms for the example distance matrix using three different linkage methods. " width="100%" /><img src="09-clustering_files/figure-html/linkageComparison-2.png" alt="Dendrograms for the example distance matrix using three different linkage methods. " width="100%" /><img src="09-clustering_files/figure-html/linkageComparison-3.png" alt="Dendrograms for the example distance matrix using three different linkage methods. " width="100%" />
<p class="caption">(\#fig:linkageComparison)Dendrograms for the example distance matrix using three different linkage methods. </p>
</div>



### Quality control
could save this example for exercises

## DBSCAN
Density-based spatial clustering of applications with noise

### Gene expression
tissue types?


## Summary

### Applications

### Strengths

### Limitations


## Exercises


Exercise solutions: \@ref(solutions-clustering)

## Extended exercises


