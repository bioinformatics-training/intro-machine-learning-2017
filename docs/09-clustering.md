# Clustering {#clustering}

<!-- Matt -->

<!-- http://www.madgroup.path.cam.ac.uk/microarraysummary.shtml -->

## Introduction

Hierarchic (produce dendrogram) vs partitioning methods


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

Graphical explanation of euclidean, manhattan and max (Chebyshev?)


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

### Example: clustering toy data sets


```r
library(RColorBrewer)
library(dendextend)
```

```
## 
## ---------------------
## Welcome to dendextend version 1.5.2
## Type citation('dendextend') for how to cite the package.
## 
## Type browseVignettes(package = 'dendextend') for the package vignette.
## The github page is: https://github.com/talgalili/dendextend/
## 
## Suggestions and bug-reports can be submitted at: https://github.com/talgalili/dendextend/issues
## Or contact: <tal.galili@gmail.com>
## 
## 	To suppress this message use:  suppressPackageStartupMessages(library(dendextend))
## ---------------------
```

```
## 
## Attaching package: 'dendextend'
```

```
## The following object is masked from 'package:ggdendro':
## 
##     theme_dendro
```

```
## The following object is masked from 'package:stats':
## 
##     cutree
```

```r
library(ggplot2)
library(GGally)

cluster_colours <- brewer.pal(8,"Dark2")

blobs <- read.csv("data/example_clusters/blobs.csv", header=F)
```


```r
aggregation <- read.table("data/example_clusters/aggregation.txt")
noisy_moons <- read.csv("data/example_clusters/noisy_moons.csv", header=F)
noisy_circles <- read.csv("data/example_clusters/noisy_circles.csv", header=F)
no_structure <- read.csv("data/example_clusters/no_structure.csv", header=F)

hclust_plots <- function(data_set, n){
  d <- dist(data_set[,1:2])
  dend <- as.dendrogram(hclust(d, method="average"))
  clusters <- cutree(dend,n,order_clusters_as_data=F)
  dend <- color_branches(dend, clusters=clusters, col=cluster_colours[1:n])
  clusters <- clusters[order(as.numeric(names(clusters)))]
  labels(dend) <- rep("", length(data_set[,1]))
  ggd <- as.ggdend(dend)
  ggd$nodes <- ggd$nodes[!(1:length(ggd$nodes[,1])),]
  plotPair <- list(ggplot(ggd),
    ggplot(data_set, aes(V1,V2)) + geom_point(col=cluster_colours[clusters], size=0.2))
  return(plotPair)
}

plotList <- c(
  hclust_plots(aggregation, 7),
  hclust_plots(noisy_moons, 2),
  hclust_plots(noisy_circles, 2),
  hclust_plots(no_structure, 3)
)

pm <- ggmatrix(
  plotList, nrow=4, ncol=2, showXAxisPlotLabels = F, showYAxisPlotLabels = F
) + theme_bw()

pm
```

\begin{figure}

{\centering \includegraphics[width=0.8\linewidth]{09-clustering_files/figure-latex/hclustToyData-1} 

}

\caption{Hierarchical clustering of toy data-sets. }(\#fig:hclustToyData)
\end{figure}

### Example: gene expression profiling of human tissues
Load required libraries

```r
library(RColorBrewer)
library(dendextend)
```

Load data

```r
load("data/tissues_gene_expression/tissuesGeneExpression.rda")
```

Inspect data

```r
table(tissue)
```

```
## tissue
##  cerebellum       colon endometrium hippocampus      kidney       liver 
##          38          34          15          31          39          26 
##    placenta 
##           6
```

```r
dim(e)
```

```
## [1] 22215   189
```

Compute distance between each sample

```r
d <- dist(t(e))
```

perform hierarchical clustering

```r
hc <- hclust(d, method="average")
plot(hc, labels=tissue, cex=0.5, hang=-1, xlab="", sub="")
```

\begin{figure}

{\centering \includegraphics[width=1\linewidth]{09-clustering_files/figure-latex/tissueDendrogram-1} 

}

\caption{Clustering of tissue samples based on gene expression profiles. }(\#fig:tissueDendrogram)
\end{figure}

use dendextend library to plot dendrogram with colour labels

```r
tissue_type <- unique(tissue)
dend <- as.dendrogram(hc)
dend_colours <- brewer.pal(length(unique(tissue)),"Dark2")
names(dend_colours) <- tissue_type
labels(dend) <- tissue[order.dendrogram(dend)]
labels_colors(dend) <- dend_colours[tissue][order.dendrogram(dend)]
labels_cex(dend) = 0.5
plot(dend, horiz=T)
```

\begin{figure}

{\centering \includegraphics[width=1\linewidth]{09-clustering_files/figure-latex/tissueDendrogramColour-1} 

}

\caption{Clustering of tissue samples based on gene expression profiles with labels coloured by tissue type. }(\#fig:tissueDendrogramColour)
\end{figure}

Define clusters by cutting tree at a specific height

```r
plot(dend, horiz=T)
abline(v=125, lwd=2, lty=2, col="blue")
```

\begin{figure}

{\centering \includegraphics[width=1\linewidth]{09-clustering_files/figure-latex/tissueDendrogramCutHeight-1} 

}

\caption{Clusters found by cutting tree at a height of 125}(\#fig:tissueDendrogramCutHeight)
\end{figure}

```r
hclusters <- cutree(dend, h=125)
table(tissue, cluster=hclusters)
```

```
##              cluster
## tissue         1  2  3  4  5  6
##   cerebellum   0 36  0  0  2  0
##   colon        0  0 34  0  0  0
##   endometrium 15  0  0  0  0  0
##   hippocampus  0 31  0  0  0  0
##   kidney      37  0  0  0  2  0
##   liver        0  0  0 24  2  0
##   placenta     0  0  0  0  0  6
```

Select a specific number of clusters.

```r
plot(dend, horiz=T)
abline(v = heights_per_k.dendrogram(dend)["8"], lwd = 2, lty = 2, col = "blue")
```

\begin{figure}

{\centering \includegraphics[width=1\linewidth]{09-clustering_files/figure-latex/tissueDendrogramEightClusters-1} 

}

\caption{Selection of eight clusters from the dendogram}(\#fig:tissueDendrogramEightClusters)
\end{figure}

```r
hclusters <- cutree(dend, k=8)
table(tissue, cluster=hclusters)
```

```
##              cluster
## tissue         1  2  3  4  5  6  7  8
##   cerebellum   0 31  0  0  2  0  5  0
##   colon        0  0 34  0  0  0  0  0
##   endometrium  0  0  0  0  0 15  0  0
##   hippocampus  0 31  0  0  0  0  0  0
##   kidney      37  0  0  0  2  0  0  0
##   liver        0  0  0 24  2  0  0  0
##   placenta     0  0  0  0  0  0  0  6
```

## Partitioning methods

### K-means

Pseudocode

to illustrate range of different types of data that can be clustered - image segmentation

\begin{figure}

{\centering \includegraphics[width=0.9\linewidth]{09-clustering_files/figure-latex/kmeansIterations-1} 

}

\caption{Iterations of the k-means algorithm}(\#fig:kmeansIterations)
\end{figure}




### DBSCAN
Density-based spatial clustering of applications with noise

### Gene expression
tissue types?


## Summary

### Applications

### Strengths

### Limitations

<!--
Not appropriate for phylogenetic analysis!!
-->

## Exercises

<!--
1. Toy clusters
2. mouse mammary time-course (kmeans and dbscan)
3. dimensionality reduction before clustering (helpful for visualization if using a partitioning method) - possibly use parasite data?
4. exercise involving heatmap

-->


Exercise solutions: \@ref(solutions-clustering)

Solutions to exercises can be found in appendix \@ref(solutions-clustering).


