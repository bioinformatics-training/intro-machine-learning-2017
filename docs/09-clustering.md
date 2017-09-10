# Clustering {#clustering}

<!-- Matt -->

<!-- http://www.madgroup.path.cam.ac.uk/microarraysummary.shtml -->

<!--
if variables are in same units - don't standardize, otherwise standardize
-->

## Introduction

What is clustering - add figure showing idea of minimizing intra-cluster variation and maximizing inter-cluster variation.




Hierarchic (produce dendrogram) vs partitioning methods

* Hierarchic agglomerative
* k-means
* DBSCAN

\begin{figure}

{\centering \includegraphics[width=0.8\linewidth]{09-clustering_files/figure-latex/clusterTypes-1} 

}

\caption{Example clusters. **A**, *blobs*; **B**, *aggregation* [@Gionis2007]; **C**, *noisy moons*; **D**, *different density*; **E**, *anisotropic distributions*; **F**, *no structure*.}(\#fig:clusterTypes)
\end{figure}

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


\begin{figure}

{\centering \includegraphics[width=0.55\linewidth]{images/hclust_demo_0} \includegraphics[width=0.55\linewidth]{images/hclust_demo_1} \includegraphics[width=0.55\linewidth]{images/hclust_demo_2} \includegraphics[width=0.55\linewidth]{images/hclust_demo_3} \includegraphics[width=0.55\linewidth]{images/hclust_demo_4} 

}

\caption{Building a dendrogram using hierarchic agglomerative clustering.}(\#fig:hierarchicClusteringDemo)
\end{figure}


Get to see clusters for all number of clusters k

### Linkage algorithms



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


Single linkage - nearest neighbours linkage
Complete linkage - furthest neighbours linkage
Average linkage - UPGMA (Unweighted Pair Group Method with Arithmetic Mean) 



<!--
Explain anatomy of the dendrogram - branches, nodes and leaves.
-->

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

\begin{figure}

{\centering \includegraphics[width=0.8\linewidth]{09-clustering_files/figure-latex/hclustBlobs-1} 

}

\caption{Hierarchical clustering of the blobs data set.}(\#fig:hclustBlobs)
\end{figure}

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
  yAxisLabels=c("aggregation", "noisy moons", "different density", "anisotropic", "no structure")
) + theme_bw()

pm
```

\begin{figure}

{\centering \includegraphics[width=0.75\linewidth]{09-clustering_files/figure-latex/hclustToyData-1} 

}

\caption{Hierarchical clustering of synthetic data-sets. }(\#fig:hclustToyData)
\end{figure}

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

\begin{figure}

{\centering \includegraphics[width=1\linewidth]{09-clustering_files/figure-latex/tissueDendrogram-1} 

}

\caption{Clustering of tissue samples based on gene expression profiles. }(\#fig:tissueDendrogram)
\end{figure}


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

\begin{figure}

{\centering \includegraphics[width=1\linewidth]{09-clustering_files/figure-latex/tissueDendrogramColour-1} 

}

\caption{Clustering of tissue samples based on gene expression profiles with labels coloured by tissue type. }(\#fig:tissueDendrogramColour)
\end{figure}

#### Defining clusters by cutting tree

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

\begin{figure}

{\centering \includegraphics[width=1\linewidth]{09-clustering_files/figure-latex/heatmapTissueExpression-1} 

}

\caption{Heatmap of the expression of the 40 genes with the highest variance.}(\#fig:heatmapTissueExpression)
\end{figure}


## K-means

### Algorithm

Pseudocode


\begin{figure}

{\centering \includegraphics[width=0.9\linewidth]{09-clustering_files/figure-latex/kmeansIterations-1} 

}

\caption{Iterations of the k-means algorithm}(\#fig:kmeansIterations)
\end{figure}

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

\begin{figure}

{\centering \includegraphics[width=1\linewidth]{09-clustering_files/figure-latex/kmeansCentreChoice-1} 

}

\caption{Initial centres determine clusters. The starting centres are shown as crosses. **A**, real clusters found; **B**, convergence to a local minimum.}(\#fig:kmeansCentreChoice)
\end{figure}
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

\begin{figure}

{\centering \includegraphics[width=1\linewidth]{09-clustering_files/figure-latex/kmeansRangeK-1} 

}

\caption{K-means clustering of the blobs data set using a range of values of k from 1-9. Cluster centres indicated with a cross.}(\#fig:kmeansRangeK)
\end{figure}


```r
tot_withinss <- sapply(k, function(i){res[[i]]$tot.withinss})
qplot(k, tot_withinss, geom=c("point", "line"), 
      ylab="Total within-cluster sum of squares") + theme_bw()
```

\begin{figure}

{\centering \includegraphics[width=0.5\linewidth]{09-clustering_files/figure-latex/choosingK-1} 

}

\caption{Variance within the clusters. Total within-cluster sum of squares plotted against k.}(\#fig:choosingK)
\end{figure}

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

\begin{figure}

{\centering \includegraphics[width=0.5\linewidth]{09-clustering_files/figure-latex/kmeansAggregationElbow-1} 

}

\caption{K-means clustering of the aggregation data set: variance within clusters.}(\#fig:kmeansAggregationElbow)
\end{figure}


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

\begin{figure}

{\centering \includegraphics[width=1\linewidth]{09-clustering_files/figure-latex/kmeansAggregationScatter-1} 

}

\caption{K-means clustering of the aggregation data set: scatterplots of clusters for k=3 and k=7. Cluster centres indicated with a cross.}(\#fig:kmeansAggregationScatter)
\end{figure}

#### Noisy moons

```r
noisy_moons <- read.csv("data/example_clusters/noisy_moons.csv", header=F)
res <- lapply(k, function(i){kmeans(noisy_moons[,1:2], i, nstart=50)})
plot_tot_withinss(res)
```

\begin{figure}

{\centering \includegraphics[width=0.5\linewidth]{09-clustering_files/figure-latex/kmeansNoisyMoonsElbow-1} 

}

\caption{K-means clustering of the noisy moons data set: variance within clusters.}(\#fig:kmeansNoisyMoonsElbow)
\end{figure}


```r
plot_clusters(noisy_moons, res, 2)
```

\begin{figure}

{\centering \includegraphics[width=0.5\linewidth]{09-clustering_files/figure-latex/kmeansNoisyMoonsScatter-1} 

}

\caption{K-means clustering of the noisy moons data set: scatterplot of clusters for k=2. Cluster centres indicated with a cross.}(\#fig:kmeansNoisyMoonsScatter)
\end{figure}

#### Different density


```r
diff_density <- as.data.frame(read.csv("data/example_clusters/different_density.csv", header=F))
res <- lapply(k, function(i){kmeans(diff_density[,1:2], i, nstart=50)})
```

```
## Warning: did not converge in 10 iterations

## Warning: did not converge in 10 iterations
```
Failure to converge, so increase number of iterations.

```r
res <- lapply(k, function(i){kmeans(diff_density[,1:2], i, iter.max=20, nstart=50)})
plot_tot_withinss(res)
```

\begin{figure}

{\centering \includegraphics[width=0.5\linewidth]{09-clustering_files/figure-latex/kmeansDiffDensityElbow-1} 

}

\caption{K-means clustering of the different density distributions data set: variance within clusters.}(\#fig:kmeansDiffDensityElbow)
\end{figure}


```r
plot_clusters(diff_density, res, 2)
```

\begin{figure}

{\centering \includegraphics[width=0.5\linewidth]{09-clustering_files/figure-latex/kmeansDiffDensityScatter-1} 

}

\caption{K-means clustering of the different density distributions data set: scatterplots of clusters for k=2 and k=3. Cluster centres indicated with a cross.}(\#fig:kmeansDiffDensityScatter)
\end{figure}

#### Anisotropic distributions

```r
aniso <- as.data.frame(read.csv("data/example_clusters/aniso.csv", header=F))
res <- lapply(k, function(i){kmeans(aniso[,1:2], i, nstart=50)})
plot_tot_withinss(res)
```

\begin{figure}

{\centering \includegraphics[width=0.5\linewidth]{09-clustering_files/figure-latex/kmeansAnisoElbow-1} 

}

\caption{K-means clustering  of the anisotropic distributions data set: variance within clusters.}(\#fig:kmeansAnisoElbow)
\end{figure}


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

\begin{figure}

{\centering \includegraphics[width=1\linewidth]{09-clustering_files/figure-latex/kmeansAnisoScatter-1} 

}

\caption{K-means clustering of the anisotropic distributions data set: scatterplots of clusters for k=2 and k=3. Cluster centres indicated with a cross.}(\#fig:kmeansAnisoScatter)
\end{figure}

#### No structure

```r
no_structure <- as.data.frame(read.csv("data/example_clusters/no_structure.csv", header=F))
res <- lapply(k, function(i){kmeans(no_structure[,1:2], i, nstart=50)})
plot_tot_withinss(res)
```

\begin{figure}

{\centering \includegraphics[width=0.5\linewidth]{09-clustering_files/figure-latex/noStructureElbow-1} 

}

\caption{K-means clustering of the data set with no structure: variance within clusters.}(\#fig:noStructureElbow)
\end{figure}


```r
plot_clusters(no_structure, res, 4)
```

\begin{figure}

{\centering \includegraphics[width=0.5\linewidth]{09-clustering_files/figure-latex/noStructureScatter-1} 

}

\caption{K-means clustering of the data set with no structure: scatterplot of clusters for k=4. Cluster centres indicated with a cross.}(\#fig:noStructureScatter)
\end{figure}

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

First we will examine the total intra-cluster variance with different values of *k*. Our data-set is fairly large, so clustering it for several values or *k* and with multiple random starting centres is computationally quite intensive. Fortunately the task readily lends itself to parallelization; we can assign the analysis of each 'k' to a different processing core. As we have seen in the previous chapters on supervised learning, [caret](http://cran.r-project.org/web/packages/caret/index.html) has parallel processing built in and we simply have to load a package for multicore processing, such as [doMC](http://cran.r-project.org/web/packages/doMC/index.html), and then register the number of cores we would like to use. Running **kmeans** in parallel is slightly more involved, but still very easy. We will start by loading [doMC](http://cran.r-project.org/web/packages/doMC/index.html) and registering all available cores:

```r
library(doMC)
```

```
## Loading required package: foreach
```

```
## Loading required package: iterators
```

```
## Loading required package: parallel
```

```r
registerDoMC()
```
To find out how many cores we have registered we can use:

```r
getDoParWorkers()
```

```
## [1] 2
```

Instead of using the **lapply** function to vectorize our code, we will instead use the parallel equivalent, **foreach**. Like **lapply**, **foreach** returns a list by default. For this example we have set a seed, rather than generate a random number, for the sake of reproducibility. Ordinarily we would omit ```set.seed(42)``` and ```.options.multicore=list(set.seed=FALSE)```.

```r
k<-1:15
set.seed(42)
res_k_15 <- foreach(
  i=k, 
  .options.multicore=list(set.seed=FALSE)) %dopar% kmeans(t(e), i, nstart=10)
plot_tot_withinss(res_k_15)
```

\begin{figure}

{\centering \includegraphics[width=1\linewidth]{09-clustering_files/figure-latex/tissueExpressionElbow-1} 

}

\caption{K-means clustering of human tissue gene expression: variance within clusters.}(\#fig:tissueExpressionElbow)
\end{figure}
<!--
set.seed(42)
res_k_15 <- lapply(k, function(i){kmeans(t(e), i, nstart=10)})
-->
There is no obvious elbow, but the rate of decrease in the total-within sum of squares appears to slow after k=5. Since we know that there are seven tissues in the data set we will try k=7. 
<!--

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
-->

```r
table(tissue, res_k_15[[7]]$cluster)
```

```
##              
## tissue         1  2  3  4  5  6  7
##   cerebellum   0  0  0  0  0  5 33
##   colon        0  0 34  0  0  0  0
##   endometrium 15  0  0  0  0  0  0
##   hippocampus  0  0  0  0 31  0  0
##   kidney      37  2  0  0  0  0  0
##   liver        0 26  0  0  0  0  0
##   placenta     0  0  0  6  0  0  0
```
The analysis has found a distinct cluster for each tissue and therefore performed slightly better than the earlier hierarchical clustering analysis, which placed endometrium and kidney observations in the same cluster.

To visualize the result in a 2D scatter plot we first need to apply dimensionality reduction. We will use principal component analysis (PCA), which was described in chapter \@ref(dimensionality-reduction).


```r
pca <- prcomp(t(e))
ggplot(data=as.data.frame(pca$x), aes(PC1,PC2)) + 
  geom_point(col=brewer.pal(7,"Dark2")[res_k_15[[7]]$cluster], 
             shape=c(49:55)[res_k_15[[7]]$cluster], size=5) + 
  theme_bw()
```

\begin{figure}

{\centering \includegraphics[width=0.5\linewidth]{09-clustering_files/figure-latex/tissueExpressionPCA-1} 

}

\caption{K-means clustering of human gene expression (k=7): scatterplot of first two principal components.}(\#fig:tissueExpressionPCA)
\end{figure}

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



\begin{figure}

{\centering \includegraphics[width=0.75\linewidth]{images/DBSCAN_Illustration} 

}

\caption{Illustration of the DBSCAN algorithm.}(\#fig:dbscanIllustration)
\end{figure}

The method requires two parameters; MinPts that is the minimum number of samples in any cluster; Eps that is the maximum distance of the sample to at least one other sample within the same cluster.

This algorithm works on a parametric approach. The two parameters involved in this algorithm are:
* e (eps) is the radius of our neighborhoods around a data point p.
* minPts is the minimum number of data points we want in a neighborhood to define a cluster.



### Implementation in R
DBSCAN is implemented in two R packages: [dbscan](https://cran.r-project.org/package=dbscan) and [fpc](https://cran.r-project.org/package=fpc). We will use the package [dbscan](https://cran.r-project.org/package=dbscan), because it is significantly faster and can handle larger data sets than [fpc](https://cran.r-project.org/package=fpc). The function has the same name in both packages and so if for any reason both packages have been loaded into our current workspace, there is a danger of calling the wrong implementation. To avoid this we can specify the package name when calling the function, e.g.:
```
dbscan::dbscan
```

We load the dbscan package in the usual way:

```r
library(dbscan)
```

### Choosing parameters
The algorithm only needs parameteres **eps** and **minPts**.

Read in data and use **kNNdist** function from [dbscan](https://cran.r-project.org/package=dbscan) package to plot the distances of the 10-nearest neighbours for each observation (figure \@ref(fig:blobsKNNdist)).


```r
blobs <- read.csv("data/example_clusters/blobs.csv", header=F)
kNNdistplot(blobs[,1:2], k=10)
abline(h=0.6)
```

\begin{figure}

{\centering \includegraphics[width=0.75\linewidth]{09-clustering_files/figure-latex/blobsKNNdist-1} 

}

\caption{10-nearest neighbour distances for the blobs data set}(\#fig:blobsKNNdist)
\end{figure}
<!-- dist2knn <- kNNdist(blobs, 10) -->


```r
res <- dbscan::dbscan(blobs[,1:2], eps=0.6, minPts = 10)
table(res$cluster)
```

```
## 
##   0   1   2   3 
##  43 484 486 487
```



```r
ggplot(blobs, aes(V1,V2)) + 
  geom_point(col=brewer.pal(8,"Dark2")[c(8,1:7)][res$cluster+1],
             shape=c(4,15,17,19)[res$cluster+1],
             size=1.5) +
  theme_bw()
```

\begin{figure}

{\centering \includegraphics[width=0.6\linewidth]{09-clustering_files/figure-latex/blobsDBSCANscatter-1} 

}

\caption{DBSCAN clustering (eps=0.6, minPts=10) of the blobs data set. Outlier observations are shown as grey crosses.}(\#fig:blobsDBSCANscatter)
\end{figure}


### Example: clustering synthetic data sets


```r
point_shapes <- c(4,15,17,19,5,6,0,1)
point_colours <- brewer.pal(8,"Dark2")[c(8,1:7)]
point_size = 1.5
center_point_size = 8

plot_dbscan_clusters <- function(data_set, dbscan_output){
  ggplot(data_set, aes(V1,V2)) + 
    geom_point(col=point_colours[dbscan_output$cluster+1],
               shape=point_shapes[dbscan_output$cluster+1], 
               size=point_size) +
    theme_bw()
}
```


#### Aggregation


```r
aggregation <- read.table("data/example_clusters/aggregation.txt")
kNNdistplot(aggregation[,1:2], k=10)
abline(h=1.8)
```

\begin{figure}

{\centering \includegraphics[width=0.75\linewidth]{09-clustering_files/figure-latex/aggregationKNNdist-1} 

}

\caption{10-nearest neighbour distances for the aggregation data set}(\#fig:aggregationKNNdist)
\end{figure}


```r
res <- dbscan::dbscan(aggregation[,1:2], eps=1.8, minPts = 10)
table(res$cluster)
```

```
## 
##   0   1   2   3   4   5   6 
##   2 168 307 105 127  45  34
```


```r
plot_dbscan_clusters(aggregation, res)
```

\begin{figure}

{\centering \includegraphics[width=0.6\linewidth]{09-clustering_files/figure-latex/aggregationDBSCANscatter-1} 

}

\caption{DBSCAN clustering (eps=1.8, minPts=10) of the aggregation data set. Outlier observations are shown as grey crosses.}(\#fig:aggregationDBSCANscatter)
\end{figure}


#### Noisy moons

```r
noisy_moons <- read.csv("data/example_clusters/noisy_moons.csv", header=F)
kNNdistplot(noisy_moons[,1:2], k=10)
abline(h=0.075)
```

\begin{figure}

{\centering \includegraphics[width=0.75\linewidth]{09-clustering_files/figure-latex/noisyMoonsKNNdist-1} 

}

\caption{10-nearest neighbour distances for the noisy moons data set}(\#fig:noisyMoonsKNNdist)
\end{figure}


```r
res <- dbscan::dbscan(noisy_moons[,1:2], eps=0.075, minPts = 10)
table(res$cluster)
```

```
## 
##   0   1   2 
##   8 748 744
```


```r
plot_dbscan_clusters(noisy_moons, res)
```

\begin{figure}

{\centering \includegraphics[width=0.6\linewidth]{09-clustering_files/figure-latex/noisyMoonsDBSCANscatter-1} 

}

\caption{DBSCAN clustering (eps=0.075, minPts=10) of the noisy moons data set. Outlier observations are shown as grey crosses.}(\#fig:noisyMoonsDBSCANscatter)
\end{figure}


#### Different density


```r
diff_density <- read.csv("data/example_clusters/different_density.csv", header=F)
kNNdistplot(diff_density[,1:2], k=10)
abline(h=0.9)
abline(h=0.6, lty=2)
```

\begin{figure}

{\centering \includegraphics[width=0.75\linewidth]{09-clustering_files/figure-latex/diffDensityKNNdist-1} 

}

\caption{10-nearest neighbour distances for the different density distributions data set}(\#fig:diffDensityKNNdist)
\end{figure}


```r
res <- dbscan::dbscan(diff_density[,1:2], eps=0.9, minPts = 10)
table(res$cluster)
```

```
## 
##    0    1 
##   40 1460
```


```r
plot_dbscan_clusters(diff_density, res)
```

\begin{figure}

{\centering \includegraphics[width=0.6\linewidth]{09-clustering_files/figure-latex/diffDensityDBSCANscatter1-1} 

}

\caption{DBSCAN clustering of the different density distribution data set with eps=0.9 and minPts=10. Outlier observations are shown as grey crosses.}(\#fig:diffDensityDBSCANscatter1)
\end{figure}


```r
res <- dbscan::dbscan(diff_density[,1:2], eps=0.6, minPts = 10)
table(res$cluster)
```

```
## 
##   0   1   2 
## 109 399 992
```


```r
plot_dbscan_clusters(diff_density, res)
```

\begin{figure}

{\centering \includegraphics[width=0.6\linewidth]{09-clustering_files/figure-latex/diffDensityDBSCANscatter2-1} 

}

\caption{DBSCAN clustering of the different density distribution data set with eps=0.6 and minPts=10. Outlier observations are shown as grey crosses.}(\#fig:diffDensityDBSCANscatter2)
\end{figure}


#### Anisotropic distributions


```r
aniso <- read.csv("data/example_clusters/aniso.csv", header=F)
kNNdistplot(aniso[,1:2], k=10)
abline(h=0.35)
```

\begin{figure}

{\centering \includegraphics[width=0.75\linewidth]{09-clustering_files/figure-latex/anisoKNNdist-1} 

}

\caption{10-nearest neighbour distances for the anisotropic distributions data set}(\#fig:anisoKNNdist)
\end{figure}


```r
res <- dbscan::dbscan(aniso[,1:2], eps=0.35, minPts = 10)
table(res$cluster)
```

```
## 
##   0   1   2   3 
##  29 489 488 494
```


```r
plot_dbscan_clusters(aniso, res)
```

\begin{figure}

{\centering \includegraphics[width=0.6\linewidth]{09-clustering_files/figure-latex/anisoDBSCANscatter-1} 

}

\caption{DBSCAN clustering (eps=0.3, minPts=10) of the anisotropic distributions data set. Outlier observations are shown as grey crosses.}(\#fig:anisoDBSCANscatter)
\end{figure}


#### No structure


```r
no_structure <- read.csv("data/example_clusters/no_structure.csv", header=F)
kNNdistplot(no_structure[,1:2], k=10)
abline(h=0.057)
```

\begin{figure}

{\centering \includegraphics[width=0.75\linewidth]{09-clustering_files/figure-latex/noStructureKNNdist-1} 

}

\caption{10-nearest neighbour distances for the data set with no structure.}(\#fig:noStructureKNNdist)
\end{figure}


```r
res <- dbscan::dbscan(no_structure[,1:2], eps=0.57, minPts = 10)
table(res$cluster)
```

```
## 
##    1 
## 1500
```

<!--No need for scatter plot-->


### Example: gene expression profiling of human tissues
Returning again to the data on gene expression of human tissues.

```r
load("data/tissues_gene_expression/tissuesGeneExpression.rda")
```


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

We'll try k=5 (default for dbscan), because there are only six observations for placenta.


```r
kNNdistplot(t(e), k=5)
abline(h=85)
```

\begin{figure}

{\centering \includegraphics[width=0.75\linewidth]{09-clustering_files/figure-latex/tissueExpressionKNNdist-1} 

}

\caption{Five-nearest neighbour distances for the gene expression profiling of human tissues data set.}(\#fig:tissueExpressionKNNdist)
\end{figure}


```r
set.seed(42)
res <- dbscan::dbscan(t(e), eps=85, minPts=5)
table(res$cluster)
```

```
## 
##  0  1  2  3  4  5  6 
## 12 37 62 34 24 15  5
```

```r
table(tissue, res$cluster)
```

```
##              
## tissue         0  1  2  3  4  5  6
##   cerebellum   2  0 31  0  0  0  5
##   colon        0  0  0 34  0  0  0
##   endometrium  0  0  0  0  0 15  0
##   hippocampus  0  0 31  0  0  0  0
##   kidney       2 37  0  0  0  0  0
##   liver        2  0  0  0 24  0  0
##   placenta     6  0  0  0  0  0  0
```


```r
pca <- prcomp(t(e))
ggplot(data=as.data.frame(pca$x), aes(PC1,PC2)) + 
  geom_point(col=brewer.pal(8,"Dark2")[c(8,1:7)][res$cluster+1], 
             shape=c(48:55)[res$cluster+1], size=5) + 
  theme_bw()
```

\begin{figure}

{\centering \includegraphics[width=0.6\linewidth]{09-clustering_files/figure-latex/tissueExpressionDBSCANscatter-1} 

}

\caption{Clustering of human tissue gene expression: scatterplot of first two principal components.}(\#fig:tissueExpressionDBSCANscatter)
\end{figure}

## Summary

### Applications

### Strengths

### Limitations

<!--
Not appropriate for phylogenetic analysis!!
-->

## Evaluating cluster quality
<!--
http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
-->

### Silhouette method

**Silhouette**
\begin{equation}
  s(i) = \frac{b(i) - a(i)}{max\left(a(i),b(i)\right)}
  (\#eq:silhouette)
\end{equation}

Method can be applied to clusters generated using any algorithm. 

### Example - k-means clustering of blobs data set
Load library required for calculating silhouette coefficients and plotting silhouettes.

```r
library(cluster)
```

We are going to take another look at k-means clustering of the blobs data-set (figure \@ref(fig:kmeansRangeK)). Specifically we are going to see if silhouette analysis supports our original choice of k=3 as the optimum number of clusters (figure \@ref(fig:choosingK)).

Silhouette analysis requires a minimum of two clusters, so we'll try values of k from 2 to 9.

```r
k <- 2:9
```
Create a palette of colours for plotting.

```r
kColours <- brewer.pal(9,"Set1")
```
Perform k-means clustering for each value of k from 2 to 9.

```r
res <- lapply(k, function(i){kmeans(blobs[,1:2], i, nstart=50)})
```

Calculate the Euclidean distance matrix

```r
d <- dist(blobs[,1:2])
```

Silhouette plot for k=2

```r
s2 <- silhouette(res[[2-1]]$cluster, d)
plot(s2, border=NA, col=kColours[sort(res[[2-1]]$cluster)], main="")
```

\begin{figure}

{\centering \includegraphics[width=0.6\linewidth]{09-clustering_files/figure-latex/silhouetteK2-1} 

}

\caption{Silhouette plot for k-means clustering of the blobs data set with k=2.}(\#fig:silhouetteK2)
\end{figure}

Silhouette plot for k=9

```r
s9 <- silhouette(res[[9-1]]$cluster, d)
plot(s9, border=NA, col=kColours[sort(res[[9-1]]$cluster)], main="")
```

\begin{figure}

{\centering \includegraphics[width=0.6\linewidth]{09-clustering_files/figure-latex/silhouetteK9-1} 

}

\caption{Silhouette plot for k-means clustering of the blobs data set with k=9.}(\#fig:silhouetteK9)
\end{figure}

Let's take a look at the silhouette plot for k=3.

```r
s3 <- silhouette(res[[3-1]]$cluster, d)
plot(s3, border=NA, col=kColours[sort(res[[3-1]]$cluster)], main="")
```

\begin{figure}

{\centering \includegraphics[width=0.6\linewidth]{09-clustering_files/figure-latex/silhouetteK3-1} 

}

\caption{Silhouette plot for k-means clustering of the blobs data set with k=3.}(\#fig:silhouetteK3)
\end{figure}

So far the silhouette plots have shown that k=3 appears to be the optimum number of clusters, but we should investigate the silhouette coefficients at other values of k. Rather than produce a silhouette plot for each value of k, we can get a useful summary by making a barplot of average silhouette coefficients.

First we will calculate the silhouette coefficient for every observation (we need to index our list of **kmeans** outputs by ```i-1```, because we are counting from k=2 ).

```r
s <- lapply(k, function(i){silhouette(res[[i-1]]$cluster, d)})
```
We can then calculate the mean silhouette coefficient for each value of k from 2 to 9.

```r
avgS <- sapply(s, function(x){mean(x[,3])})
```
Now we have the data we need to produce a barplot.

```r
dat <- as.data.frame(cbind(k, avgS))
ggplot(data=dat, aes(x=k, y=avgS)) + 
         geom_bar(stat="identity", fill="steelblue") +
  geom_text(aes(label=round(avgS,2)), vjust=1.6, color="white", size=3.5)+
  labs(y="Average silhouette coefficient") +
  scale_x_continuous(breaks=2:9) +
  theme_bw()
```

\begin{figure}

{\centering \includegraphics[width=0.75\linewidth]{09-clustering_files/figure-latex/silhouetteAllK-1} 

}

\caption{Barplot of the average silhouette coefficients resulting from k-means clustering of the blobs data-set using values of k from 1-9.}(\#fig:silhouetteAllK)
\end{figure}

The bar plot (figure \@ref(fig:silhouetteAllK)) confirms that the optimum number of clusters is three.










## Exercises

<!--
1. Toy/synthetic clusters
2. mouse mammary time-course (kmeans and dbscan)
3. dimensionality reduction before clustering (helpful for visualization if using a partitioning method) - possibly use parasite data?
4. exercise involving heatmap

-->


Exercise solutions: \@ref(solutions-clustering)

Solutions to exercises can be found in appendix \@ref(solutions-clustering).


