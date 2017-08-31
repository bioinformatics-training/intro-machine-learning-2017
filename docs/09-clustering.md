# Clustering {#clustering}

<!-- Matt -->

<!-- http://www.madgroup.path.cam.ac.uk/microarraysummary.shtml -->

## Introduction

Hierarchic (produce dendrogram) vs partitioning methods


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

Graphical explanation of euclidean, manhattan and max (Chebyshev?)


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
  plotList, nrow=4, ncol=2, showXAxisPlotLabels = F, showYAxisPlotLabels = F, xAxisLabels=c("dendrogram", "scatter plot"), yAxisLabels=c("aggregation", "noisy moons", "noisy circles", "no structure")
) + theme_bw()

pm
```

<div class="figure" style="text-align: center">
<img src="09-clustering_files/figure-html/hclustToyData-1.png" alt="Hierarchical clustering of toy data-sets. " width="75%" />
<p class="caption">(\#fig:hclustToyData)Hierarchical clustering of toy data-sets. </p>
</div>

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

<div class="figure" style="text-align: center">
<img src="09-clustering_files/figure-html/tissueDendrogram-1.png" alt="Clustering of tissue samples based on gene expression profiles. " width="100%" />
<p class="caption">(\#fig:tissueDendrogram)Clustering of tissue samples based on gene expression profiles. </p>
</div>

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

<div class="figure" style="text-align: center">
<img src="09-clustering_files/figure-html/tissueDendrogramColour-1.png" alt="Clustering of tissue samples based on gene expression profiles with labels coloured by tissue type. " width="100%" />
<p class="caption">(\#fig:tissueDendrogramColour)Clustering of tissue samples based on gene expression profiles with labels coloured by tissue type. </p>
</div>

Define clusters by cutting tree at a specific height

```r
plot(dend, horiz=T)
abline(v=125, lwd=2, lty=2, col="blue")
```

<div class="figure" style="text-align: center">
<img src="09-clustering_files/figure-html/tissueDendrogramCutHeight-1.png" alt="Clusters found by cutting tree at a height of 125" width="100%" />
<p class="caption">(\#fig:tissueDendrogramCutHeight)Clusters found by cutting tree at a height of 125</p>
</div>

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

<div class="figure" style="text-align: center">
<img src="09-clustering_files/figure-html/tissueDendrogramEightClusters-1.png" alt="Selection of eight clusters from the dendogram" width="100%" />
<p class="caption">(\#fig:tissueDendrogramEightClusters)Selection of eight clusters from the dendogram</p>
</div>

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

#### Algorithm

Pseudocode

to illustrate range of different types of data that can be clustered - image segmentation

<div class="figure" style="text-align: center">
<img src="09-clustering_files/figure-html/kmeansIterations-1.png" alt="Iterations of the k-means algorithm" width="90%" />
<p class="caption">(\#fig:kmeansIterations)Iterations of the k-means algorithm</p>
</div>

The default setting of the **kmeans** function is to perform a maximum of 10 iterations and if the algorithm fails to converge a warning is issued. The maximum number of iterations is set with the argument **iter.max**.

#### Choosing initial cluster centres

```r
library(RColorBrewer)
point_shapes <- c(15,17,19)
point_colours <- brewer.pal(3,"Dark2")
point_size = 1.5
center_point_size = 8

blobs <- as.data.frame(read.csv("data/example_clusters/blobs.csv", header=F))

good_centres <- as.data.frame(matrix(c(2,8,7,3,12,7), ncol=2, byrow=T))
bad_centres <- as.data.frame(matrix(c(13,13,8,12,2,2), ncol=2, byrow=T))

good_result <- kmeans(blobs[,1:2], centers=good_centres)
bad_result <- kmeans(blobs[,1:2], centers=bad_centres)

plotList <- list(
ggplot(blobs, aes(V1,V2)) + geom_point(col=point_colours[good_result$cluster], shape=point_shapes[good_result$cluster], size=point_size) + geom_point(data=good_centres, aes(V1,V2), shape=3, col="black", size=center_point_size) + theme_bw(),
ggplot(blobs, aes(V1,V2)) + geom_point(col=point_colours[bad_result$cluster], shape=point_shapes[bad_result$cluster], size=point_size) + geom_point(data=bad_centres, aes(V1,V2), shape=3, col="black", size=center_point_size) + theme_bw()
)

pm <- ggmatrix(
  plotList, nrow=1, ncol=2, showXAxisPlotLabels = T, showYAxisPlotLabels = T, xAxisLabels=c("A", "B")
) + theme_bw()

pm
```

<div class="figure" style="text-align: center">
<img src="09-clustering_files/figure-html/kmeansCentreChoice-1.png" alt="Initial centres determine clusters. The starting centres are shown as crosses. **A**, real clusters found; **B**, convergence to a local minimum." width="100%" />
<p class="caption">(\#fig:kmeansCentreChoice)Initial centres determine clusters. The starting centres are shown as crosses. **A**, real clusters found; **B**, convergence to a local minimum.</p>
</div>
Convergence to a local minimum can be avoided by starting the algorithm multiple times, with different random centres. The **nstart** argument to the **k-means** function can be used to specify the number of random sets and optimal solution will be selected automatically.


#### Choosing k


```r
cluster_colours <- brewer.pal(9,"Set1")
k <- 1:9
res <- lapply(k, function(i){kmeans(blobs[,1:2], i, nstart=50)})
tot_withinss <- sapply(k, function(i){res[[i]]$tot.withinss})

plotList <- lapply(k, function(i){
  ggplot(blobs, aes(V1, V2)) + 
    geom_point(col=cluster_colours[res[[i]]$cluster], size=1) +
    geom_point(data=as.data.frame(res[[i]]$centers), aes(V1,V2), shape=3, col="black", size=5) +
    annotate("text", x=2, y=13, label=paste("k=", i, sep=""), size=8, col="black") +
    theme_bw()
}
)

pm <- ggmatrix(
  plotList, nrow=3, ncol=3, showXAxisPlotLabels = T, showYAxisPlotLabels = T
) + theme_bw()

pm
```

<div class="figure" style="text-align: center">
<img src="09-clustering_files/figure-html/kmeansRangeK-1.png" alt="K-means clustering of the blobs data set using a range of values of k from 1-9. Cluster centres indicated with a cross." width="100%" />
<p class="caption">(\#fig:kmeansRangeK)K-means clustering of the blobs data set using a range of values of k from 1-9. Cluster centres indicated with a cross.</p>
</div>

*N.B.* we have set ```nstart=50``` so that the algorithm is started 50 times wi

### DBSCAN
Density-based spatial clustering of applications with noise

#### Algorithm


#### Choosing parameters



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


