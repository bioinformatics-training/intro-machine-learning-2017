# Clustering

<!-- Matt -->

<!-- http://www.madgroup.path.cam.ac.uk/microarraysummary.shtml -->

## Introduction

## Types of cluster

\begin{figure}

{\centering \includegraphics[width=0.8\linewidth]{09-clustering_files/figure-latex/clusterTypes-1} 

}

\caption{Example clusters. **A**, *blobs*; **B**, *aggregation* [@Gionis2007]; **C**, *noisy moons*; **D**, *noisy circles*; **E**, *D31* [@Veenman2002]; **F**, *no structure*.}(\#fig:clusterTypes)
\end{figure}

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



\begin{table}

\caption{(\#tab:distance-matrix)Example distance matrix}
\centering
\begin{tabular}[t]{lllll}
\toprule
  & A & B & C & D\\
\midrule
B & 2 &  &  & \\
C & 6 & 5 &  & \\
D & 10 & 10 & 5 & \\
E & 9 & 8 & 3 & 4\\
\bottomrule
\end{tabular}
\end{table}

### Linkage algorithms
Make one section
panel of three dendrograms
one table

Single linkage - nearest neighbours linkage
Complete linkage - furthest neighbours linkage
Average linkage - UPGMA (Unweighted Pair Group Method with Arithmetic Mean) 





\begin{table}

\caption{(\#tab:distance-merge)Merge distances for objects in the example distance matrix using three different linkage methods.}
\centering
\begin{tabular}[t]{llll}
\toprule
Groups & Single & Complete & Average\\
\midrule
A,B,C,D,E & 0 & 0 & 0\\
(A,B),C,D,E & 2 & 2 & 2\\
(A,B),(C,E),D & 3 & 3 & 3\\
(A,B)(C,D,E) & 4 & 5 & 4.5\\
(A,B,C,D,E) & 5 & 10 & 8\\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}

{\centering \includegraphics[width=1\linewidth]{09-clustering_files/figure-latex/linkageComparison-1} \includegraphics[width=1\linewidth]{09-clustering_files/figure-latex/linkageComparison-2} \includegraphics[width=1\linewidth]{09-clustering_files/figure-latex/linkageComparison-3} 

}

\caption{Dendrograms for the example distance matrix using three different linkage methods. }(\#fig:linkageComparison)
\end{figure}



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


