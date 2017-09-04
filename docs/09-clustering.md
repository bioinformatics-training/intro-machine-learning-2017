# Clustering {#clustering}

<!-- Matt -->

<!-- http://www.madgroup.path.cam.ac.uk/microarraysummary.shtml -->

## Introduction

What is clustering - add figure showing idea of minimizing intra-cluster variation and maximizing inter-cluster variation.




Hierarchic (produce dendrogram) vs partitioning methods

* Hierarchic agglomerative
* k-means
* DBSCAN

<div class="figure" style="text-align: center">
<img src="09-clustering_files/figure-html/clusterTypes-1.png" alt="Example clusters. **A**, *blobs*; **B**, *aggregation* [@Gionis2007]; **C**, *noisy moons*; **D**, *different density*; **E**, *anisotropic distributions*; **F**, *no structure*." width="80%" />
<p class="caption">(\#fig:clusterTypes)Example clusters. **A**, *blobs*; **B**, *aggregation* [@Gionis2007]; **C**, *noisy moons*; **D**, *different density*; **E**, *anisotropic distributions*; **F**, *no structure*.</p>
</div>

## Distance metrics

dist function
cor as.dist(1-cor(x))

**Minkowski distance:**
\begin{equation}
  distance\left(x,y,p\right)=\left(\sum_{i=1}^{n} abs(x_i-y_i)^p\right)^{1/p}
  (\#eq:minkowski)
\end{equation}

Graphical explanation of euclidean, manhattan and max (Chebyshev?)


### Image segmentation



## Hierarchic agglomerative




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



<!--
Explain anatomy of the dendrogram - branches, nodes and leaves.
-->


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

### Example: clustering synthetic data sets

#### Step-by-step instructions
1. Load required packages.

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
```

2. Retrieve a palette of eight colours.

```r
cluster_colours <- brewer.pal(8,"Dark2")
```

3. Read in data for **blobs** example.

```r
blobs <- read.csv("data/example_clusters/blobs.csv", header=F)
```

4. Create distance matrix using Euclidean distance metric.

```r
d <- dist(blobs[,1:2])
```

5. Perform hierarchical clustering using the **average** agglomeration method and convert the result to an object of class **dendrogram**. A **dendrogram** object can be edited using the advanced features of the **dendextend** package.

```r
dend <- as.dendrogram(hclust(d, method="average"))
```

6. Cut the tree into three clusters

```r
clusters <- cutree(dend,3,order_clusters_as_data=F)
```

7. The vector **clusters** contains the cluster membership (in this case *1*, *2* or *3*) of each observation (data point) in the order they appear on the dendrogram. We can use this vector to colour the branches of the dendrogram by cluster.

```r
dend <- color_branches(dend, clusters=clusters, col=cluster_colours[1:3])
```

8. We can use the **labels** function to annotate the leaves of the dendrogram. However, it is not possible to create legible labels for the 1,500 leaves in our example dendrogram, so we will set the label for each leaf to an empty string.

```r
labels(dend) <- rep("", length(blobs[,1]))
```

9. If we want to plot the dendrogram using **ggplot**, we must convert it to an object of class **ggdend**.

```r
ggd <- as.ggdend(dend)
```

10. The **nodes** attribute of **ggd** is a data.frame of parameters related to the plotting of dendogram nodes. The **nodes** data.frame contains some NAs which will generate warning messages when **ggd** is processed by **ggplot**. Since we are not interested in annotating dendrogram nodes, the easiest option here is to delete all of the rows of **nodes**.

```r
ggd$nodes <- ggd$nodes[!(1:length(ggd$nodes[,1])),]
```

11. We can use the cluster membership of each observation contained in the vector **clusters** to assign colours to the data points of a scatterplot. However, first we need to reorder the vector so that the cluster memberships are in the same order that the observations appear in the data.frame of observations. Fortunately the names of the elements of the vector are the indices of the observations in the data.frame and so reordering can be accomplished in one line.

```r
clusters <- clusters[order(as.numeric(names(clusters)))]
```

12. We are now ready to plot a dendrogram and scatterplot. We will use the **ggmatrix** function from the **GGally** package to place the plots side-by-side. 


```r
plotList <- list(ggplot(ggd),
                 ggplot(blobs, aes(V1,V2)) + 
                   geom_point(col=cluster_colours[clusters], size=0.2)
                 )

pm <- ggmatrix(
  plotList, nrow=1, ncol=2, showXAxisPlotLabels = F, showYAxisPlotLabels = F, 
  xAxisLabels=c("dendrogram", "scatter plot")
) + theme_bw()

pm
```

<div class="figure" style="text-align: center">
<img src="09-clustering_files/figure-html/hclustBlobs-1.png" alt="Hierarchical clustering of the blobs data set." width="80%" />
<p class="caption">(\#fig:hclustBlobs)Hierarchical clustering of the blobs data set.</p>
</div>

#### Clustering of other synthetic data sets


```r
aggregation <- read.table("data/example_clusters/aggregation.txt")
noisy_moons <- read.csv("data/example_clusters/noisy_moons.csv", header=F)
diff_density <- read.csv("data/example_clusters/different_density.csv", header=F)
aniso <- read.csv("data/example_clusters/aniso.csv", header=F)
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
    ggplot(data_set, aes(V1,V2)) + 
      geom_point(col=cluster_colours[clusters], size=0.2))
  return(plotPair)
}

plotList <- c(
  hclust_plots(aggregation, 7),
  hclust_plots(noisy_moons, 2),
  hclust_plots(diff_density, 2),
  hclust_plots(aniso, 3),
  hclust_plots(no_structure, 3)
)

pm <- ggmatrix(
  plotList, nrow=5, ncol=2, showXAxisPlotLabels = F, showYAxisPlotLabels = F,
  xAxisLabels=c("dendrogram", "scatter plot"), 
  yAxisLabels=c("aggregation", "noisy moons", "diff. density", "anisotropic", "no structure")
) + theme_bw()

pm
```

<div class="figure" style="text-align: center">
<img src="09-clustering_files/figure-html/hclustToyData-1.png" alt="Hierarchical clustering of synthetic data-sets. " width="75%" />
<p class="caption">(\#fig:hclustToyData)Hierarchical clustering of synthetic data-sets. </p>
</div>

### Example: gene expression profiling of human tissues

#### Basics
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


#### Colour labels

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

#### Defining clusters by cutting tree

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

#### Heatmap
Base R provides a **heatmap** function, but we will use the more advanced **heatmap.2** from the **gplots** package.

```r
library(gplots)
```

```
## 
## Attaching package: 'gplots'
```

```
## The following object is masked from 'package:stats':
## 
##     lowess
```

Define a colour palette (also known as a lookup table).

```r
heatmap_colours <- colorRampPalette(brewer.pal(9, "PuBuGn"))(100)
```

Calculate the variance of each gene.

```r
geneVariance <- apply(e,1,var)
```

Find the row numbers of the 40 genes with the highest variance.

```r
idxTop40 <- order(-geneVariance)[1:40]
```

Define colours for tissues.

```r
tissueColours <- palette(brewer.pal(8, "Dark2"))[as.numeric(as.factor(tissue))]
```

Plot heatmap.

```r
heatmap.2(e[idxTop40,], labCol=tissue, trace="none",
          ColSideColors=tissueColours, col=heatmap_colours)
```

<div class="figure" style="text-align: center">
<img src="09-clustering_files/figure-html/heatmapTissueExpression-1.png" alt="Heatmap of the expression of the 40 genes with the highest variance." width="100%" />
<p class="caption">(\#fig:heatmapTissueExpression)Heatmap of the expression of the 40 genes with the highest variance.</p>
</div>


## K-means

### Algorithm

Pseudocode

to illustrate range of different types of data that can be clustered - image segmentation

<div class="figure" style="text-align: center">
<img src="09-clustering_files/figure-html/kmeansIterations-1.png" alt="Iterations of the k-means algorithm" width="90%" />
<p class="caption">(\#fig:kmeansIterations)Iterations of the k-means algorithm</p>
</div>

The default setting of the **kmeans** function is to perform a maximum of 10 iterations and if the algorithm fails to converge a warning is issued. The maximum number of iterations is set with the argument **iter.max**.

### Choosing initial cluster centres

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
ggplot(blobs, aes(V1,V2)) + 
  geom_point(col=point_colours[good_result$cluster], shape=point_shapes[good_result$cluster], 
             size=point_size) + 
  geom_point(data=good_centres, aes(V1,V2), shape=3, col="black", size=center_point_size) + 
  theme_bw(),
ggplot(blobs, aes(V1,V2)) + 
  geom_point(col=point_colours[bad_result$cluster], shape=point_shapes[bad_result$cluster], 
             size=point_size) + 
  geom_point(data=bad_centres, aes(V1,V2), shape=3, col="black", size=center_point_size) + 
  theme_bw()
)

pm <- ggmatrix(
  plotList, nrow=1, ncol=2, showXAxisPlotLabels = T, showYAxisPlotLabels = T, 
  xAxisLabels=c("A", "B")
) + theme_bw()

pm
```

<div class="figure" style="text-align: center">
<img src="09-clustering_files/figure-html/kmeansCentreChoice-1.png" alt="Initial centres determine clusters. The starting centres are shown as crosses. **A**, real clusters found; **B**, convergence to a local minimum." width="100%" />
<p class="caption">(\#fig:kmeansCentreChoice)Initial centres determine clusters. The starting centres are shown as crosses. **A**, real clusters found; **B**, convergence to a local minimum.</p>
</div>
Convergence to a local minimum can be avoided by starting the algorithm multiple times, with different random centres. The **nstart** argument to the **k-means** function can be used to specify the number of random sets and optimal solution will be selected automatically.


### Choosing k


```r
point_colours <- brewer.pal(9,"Set1")
k <- 1:9
res <- lapply(k, function(i){kmeans(blobs[,1:2], i, nstart=50)})

plotList <- lapply(k, function(i){
  ggplot(blobs, aes(V1, V2)) + 
    geom_point(col=point_colours[res[[i]]$cluster], size=1) +
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


```r
tot_withinss <- sapply(k, function(i){res[[i]]$tot.withinss})
qplot(k, tot_withinss, geom=c("point", "line"), 
      ylab="Total within-cluster sum of squares") + theme_bw()
```

<div class="figure" style="text-align: center">
<img src="09-clustering_files/figure-html/choosingK-1.png" alt="Variance within the clusters. Total within-cluster sum of squares plotted against k." width="50%" />
<p class="caption">(\#fig:choosingK)Variance within the clusters. Total within-cluster sum of squares plotted against k.</p>
</div>

*N.B.* we have set ```nstart=50``` to run the algorithm 50 times, starting from different, random sets of centroids.


### Example: clustering synthetic data sets
Let's see how k-means performs on the other toy data sets. First we will define some variables and functions we will use in the analysis of all data sets.

```r
k=1:9
point_shapes <- c(15,17,19,5,6,0,1)
point_colours <- brewer.pal(7,"Dark2")
point_size = 1.5
center_point_size = 8

plot_tot_withinss <- function(kmeans_output){
  tot_withinss <- sapply(k, function(i){kmeans_output[[i]]$tot.withinss})
  qplot(k, tot_withinss, geom=c("point", "line"), 
        ylab="Total within-cluster sum of squares") + theme_bw()
}

plot_clusters <- function(data_set, kmeans_output, num_clusters){
    ggplot(data_set, aes(V1,V2)) + 
    geom_point(col=point_colours[kmeans_output[[num_clusters]]$cluster],
               shape=point_shapes[kmeans_output[[num_clusters]]$cluster], 
               size=point_size) +
    geom_point(data=as.data.frame(kmeans_output[[num_clusters]]$centers), aes(V1,V2),
               shape=3,col="black",size=center_point_size) + 
    theme_bw()
}
```

#### Aggregation

```r
aggregation <- as.data.frame(read.table("data/example_clusters/aggregation.txt"))
res <- lapply(k, function(i){kmeans(aggregation[,1:2], i, nstart=50)})
plot_tot_withinss(res)
```

<div class="figure" style="text-align: center">
<img src="09-clustering_files/figure-html/kmeansAggregationElbow-1.png" alt="K-means clustering of the aggregation data set: variance within clusters." width="50%" />
<p class="caption">(\#fig:kmeansAggregationElbow)K-means clustering of the aggregation data set: variance within clusters.</p>
</div>


```r
plotList <- list(
  plot_clusters(aggregation, res, 3),
  plot_clusters(aggregation, res, 7)
)
pm <- ggmatrix(
  plotList, nrow=1, ncol=2, showXAxisPlotLabels = T, showYAxisPlotLabels = T, 
  xAxisLabels=c("k=3", "k=7")
) + theme_bw()
pm
```

<div class="figure" style="text-align: center">
<img src="09-clustering_files/figure-html/kmeansAggregationScatter-1.png" alt="K-means clustering of the aggregation data set: scatterplots of clusters for k=3 and k=7. Cluster centres indicated with a cross." width="100%" />
<p class="caption">(\#fig:kmeansAggregationScatter)K-means clustering of the aggregation data set: scatterplots of clusters for k=3 and k=7. Cluster centres indicated with a cross.</p>
</div>

#### Noisy moons

```r
noisy_moons <- read.csv("data/example_clusters/noisy_moons.csv", header=F)
res <- lapply(k, function(i){kmeans(noisy_moons[,1:2], i, nstart=50)})
plot_tot_withinss(res)
```

<div class="figure" style="text-align: center">
<img src="09-clustering_files/figure-html/kmeansNoisyMoonsElbow-1.png" alt="K-means clustering of the noisy moons data set: variance within clusters." width="50%" />
<p class="caption">(\#fig:kmeansNoisyMoonsElbow)K-means clustering of the noisy moons data set: variance within clusters.</p>
</div>


```r
plot_clusters(noisy_moons, res, 2)
```

<div class="figure" style="text-align: center">
<img src="09-clustering_files/figure-html/kmeansNoisyMoonsScatter-1.png" alt="K-means clustering of the noisy moons data set: scatterplot of clusters for k=2. Cluster centres indicated with a cross." width="50%" />
<p class="caption">(\#fig:kmeansNoisyMoonsScatter)K-means clustering of the noisy moons data set: scatterplot of clusters for k=2. Cluster centres indicated with a cross.</p>
</div>

#### Different density

```r
diff_density <- as.data.frame(read.csv("data/example_clusters/different_density.csv", header=F))
res <- lapply(k, function(i){kmeans(diff_density[,1:2], i, nstart=50)})
```

```
## Warning: did not converge in 10 iterations

## Warning: did not converge in 10 iterations

## Warning: did not converge in 10 iterations

## Warning: did not converge in 10 iterations
```

```r
plot_tot_withinss(res)
```

<div class="figure" style="text-align: center">
<img src="09-clustering_files/figure-html/kmeansDiffDensityElbow-1.png" alt="K-means clustering of the different density distributions data set: variance within clusters." width="50%" />
<p class="caption">(\#fig:kmeansDiffDensityElbow)K-means clustering of the different density distributions data set: variance within clusters.</p>
</div>


```r
plot_clusters(diff_density, res, 2)
```

<div class="figure" style="text-align: center">
<img src="09-clustering_files/figure-html/kmeansDiffDensityScatter-1.png" alt="K-means clustering of the different density distributions data set: scatterplots of clusters for k=2 and k=3. Cluster centres indicated with a cross." width="50%" />
<p class="caption">(\#fig:kmeansDiffDensityScatter)K-means clustering of the different density distributions data set: scatterplots of clusters for k=2 and k=3. Cluster centres indicated with a cross.</p>
</div>

#### Anisotropic distributions

```r
aniso <- as.data.frame(read.csv("data/example_clusters/aniso.csv", header=F))
res <- lapply(k, function(i){kmeans(aniso[,1:2], i, nstart=50)})
plot_tot_withinss(res)
```

<div class="figure" style="text-align: center">
<img src="09-clustering_files/figure-html/kmeansAnisoElbow-1.png" alt="K-means clustering  of the anisotropic distributions data set: variance within clusters." width="50%" />
<p class="caption">(\#fig:kmeansAnisoElbow)K-means clustering  of the anisotropic distributions data set: variance within clusters.</p>
</div>


```r
plotList <- list(
  plot_clusters(aniso, res, 2),
  plot_clusters(aniso, res, 3)
)
pm <- ggmatrix(
  plotList, nrow=1, ncol=2, showXAxisPlotLabels = T, 
  showYAxisPlotLabels = T, xAxisLabels=c("k=2", "k=3")
) + theme_bw()
pm
```

<div class="figure" style="text-align: center">
<img src="09-clustering_files/figure-html/kmeansAnisoScatter-1.png" alt="K-means clustering of the anisotropic distributions data set: scatterplots of clusters for k=2 and k=3. Cluster centres indicated with a cross." width="100%" />
<p class="caption">(\#fig:kmeansAnisoScatter)K-means clustering of the anisotropic distributions data set: scatterplots of clusters for k=2 and k=3. Cluster centres indicated with a cross.</p>
</div>

#### No structure

```r
no_structure <- as.data.frame(read.csv("data/example_clusters/no_structure.csv", header=F))
res <- lapply(k, function(i){kmeans(no_structure[,1:2], i, nstart=50)})
plot_tot_withinss(res)
```

<div class="figure" style="text-align: center">
<img src="09-clustering_files/figure-html/noStructureElbow-1.png" alt="K-means clustering of the data set with no structure: variance within clusters." width="50%" />
<p class="caption">(\#fig:noStructureElbow)K-means clustering of the data set with no structure: variance within clusters.</p>
</div>


```r
plot_clusters(no_structure, res, 4)
```

<div class="figure" style="text-align: center">
<img src="09-clustering_files/figure-html/noStructureScatter-1.png" alt="K-means clustering of the data set with no structure: scatterplot of clusters for k=4. Cluster centres indicated with a cross." width="50%" />
<p class="caption">(\#fig:noStructureScatter)K-means clustering of the data set with no structure: scatterplot of clusters for k=4. Cluster centres indicated with a cross.</p>
</div>

### Example: gene expression profiling of human tissues
Let's return to the data on gene expression of human tissues.
Load data

```r
load("data/tissues_gene_expression/tissuesGeneExpression.rda")
```

As we saw earlier, the data set contains expression levels for over 22,000 transcripts in seven tissues.

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

First we will examine the total intra-cluster variance with different values of *k*. In practice we would set **nstart** to a large value (e.g. 50), but in the interests of speed for this demonstration we will set it to one. We use **set.seed** to make this example reproducible, but in practice you would allow **R** to generate a random seed.


```r
k<-1:15
set.seed(42)
res_k_15 <- lapply(k, function(i){kmeans(t(e), i, nstart=1)})
plot_tot_withinss(res_k_15)
```

<div class="figure" style="text-align: center">
<img src="09-clustering_files/figure-html/tissueExpressionElbow-1.png" alt="K-means clustering of human tissue gene expression: variance within clusters." width="100%" />
<p class="caption">(\#fig:tissueExpressionElbow)K-means clustering of human tissue gene expression: variance within clusters.</p>
</div>
If we had set **nstart** to a higher value we would have obtained a smoother curve in figure \@ref(fig:tissueExpressionElbow). There is no obvious elbow, but the rate of decrease in the total-within sum of squares appears to slow after k=5. Since we know that there are seven tissues in the data set we will try k=7. 


```r
set.seed(42)
res <- kmeans(t(e), 7, nstart=10)
table(tissue, res$cluster)
```

```
##              
## tissue         1  2  3  4  5  6  7
##   cerebellum   0  0  0 33  0  0  5
##   colon        0  0  0  0  0 34  0
##   endometrium  0  0  0  0 15  0  0
##   hippocampus  0  0  0  0  0  0 31
##   kidney       0  0 39  0  0  0  0
##   liver       26  0  0  0  0  0  0
##   placenta     0  6  0  0  0  0  0
```
The analysis has found a distinct cluster for each tissue and therefore performed slightly better than the earlier hierarchical clustering analysis, which placed endometrium and kidney observations in the same cluster.

To visualize the result in a 2D scatter plot we first need to apply dimensionality reduction. We will use principal component analysis (PCA), which was described in chapter \@ref(dimensionality-reduction).


```r
pca <- prcomp(t(e))
ggplot(data=as.data.frame(pca$x), aes(PC1,PC2)) + 
  geom_point(col=brewer.pal(7,"Dark2")[res$cluster], 
             shape=c(49:55)[res$cluster], size=5) + 
  theme_bw()
```

<div class="figure" style="text-align: center">
<img src="09-clustering_files/figure-html/tissueExpressionPCA-1.png" alt="K-means clustering of human gene expression (k=7): scatterplot of first two principal components." width="50%" />
<p class="caption">(\#fig:tissueExpressionPCA)K-means clustering of human gene expression (k=7): scatterplot of first two principal components.</p>
</div>

## DBSCAN
Density-based spatial clustering of applications with noise

### Algorithm


Abstract DBSCAN algorithm in pseudocode [@Schubert2017]

```
1 Compute neighbours of each point and identify core points   // Identify core points
2 Join neighbouring core points into clusters                 // Assign core points
3 foreach non-core point do
      Add to a neighbouring core point if possible            // Assign border points
      Otherwise, add to noise                                 // Assign noise points
```



<div class="figure" style="text-align: center">
<img src="images/DBSCAN_Illustration.svg" alt="Illustration of the DBSCAN algorithm." width="75%" />
<p class="caption">(\#fig:dbscanIllustration)Illustration of the DBSCAN algorithm.</p>
</div>



### Choosing parameters
The algorithm only needs parameteres **eps** and **minPts**.




```r
library(dbscan)
```


```r
blobs <- read.csv("data/example_clusters/blobs.csv", header=F)
dist2knn <- kNNdist(blobs, 3)
```

<!--
?dbscan::dbscan
aggregation <- read.table("data/example_clusters/aggregation.txt")
noisy_moons <- read.csv("data/example_clusters/noisy_moons.csv", header=F)
noisy_circles <- read.csv("data/example_clusters/noisy_circles.csv", header=F)
aniso <- read.csv("data/example_clusters/aniso.csv", header=F)
no_structure <- read.csv("data/example_clusters/no_structure.csv", header=F)
-->


### Example: clustering synthetic data sets


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
1. Toy/synthetic clusters
2. mouse mammary time-course (kmeans and dbscan)
3. dimensionality reduction before clustering (helpful for visualization if using a partitioning method) - possibly use parasite data?
4. exercise involving heatmap

-->


Exercise solutions: \@ref(solutions-clustering)

Solutions to exercises can be found in appendix \@ref(solutions-clustering).


