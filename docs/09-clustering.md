# Clustering

<!-- Matt -->

<!-- http://www.madgroup.path.cam.ac.uk/microarraysummary.shtml -->

## Introduction

## Types of cluster

<div class="figure" style="text-align: center">
<img src="09-clustering_files/figure-html/clusterTypes-1.png" alt="Example clusters" width="80%" />
<p class="caption">(\#fig:clusterTypes)Example clusters</p>
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

#### Single linkage
nearest  neighbours linkage



Table: (\#tab:single-merge)Merge distances for single linkage.

Distance   Groups        
---------  --------------
0          A,B,C,D,E     
2          (A,B),C,D,E   
3          (A,B),(C,E),D 
4          (A,B)(C,D,E)  
5          (A,B,C,D,E)   


#### Complete linkage
furthest neighbours




#### Average linkage
UPGMA (Unweighted Pair Group Method with Arithmetic Mean) 




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


