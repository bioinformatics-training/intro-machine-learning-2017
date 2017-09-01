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

dist function
cor as.dist(1-cor(x))

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

### Example: clustering toy data sets

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
                 ggplot(blobs, aes(V1,V2)) + geom_point(col=cluster_colours[clusters], size=0.2)
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

#### Clustering of other toy data sets


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

\begin{figure}

{\centering \includegraphics[width=0.75\linewidth]{09-clustering_files/figure-latex/hclustToyData-1} 

}

\caption{Hierarchical clustering of toy data-sets. }(\#fig:hclustToyData)
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


## Partitioning methods

### K-means

#### Algorithm

Pseudocode

to illustrate range of different types of data that can be clustered - image segmentation

\begin{figure}

{\centering \includegraphics[width=0.9\linewidth]{09-clustering_files/figure-latex/kmeansIterations-1} 

}

\caption{Iterations of the k-means algorithm}(\#fig:kmeansIterations)
\end{figure}

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

\begin{figure}

{\centering \includegraphics[width=1\linewidth]{09-clustering_files/figure-latex/kmeansCentreChoice-1} 

}

\caption{Initial centres determine clusters. The starting centres are shown as crosses. **A**, real clusters found; **B**, convergence to a local minimum.}(\#fig:kmeansCentreChoice)
\end{figure}
Convergence to a local minimum can be avoided by starting the algorithm multiple times, with different random centres. The **nstart** argument to the **k-means** function can be used to specify the number of random sets and optimal solution will be selected automatically.


#### Choosing k


```r
cluster_colours <- brewer.pal(9,"Set1")
k <- 1:9
res <- lapply(k, function(i){kmeans(blobs[,1:2], i, nstart=50)})

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

\begin{figure}

{\centering \includegraphics[width=1\linewidth]{09-clustering_files/figure-latex/kmeansRangeK-1} 

}

\caption{K-means clustering of the blobs data set using a range of values of k from 1-9. Cluster centres indicated with a cross.}(\#fig:kmeansRangeK)
\end{figure}


```r
tot_withinss <- sapply(k, function(i){res[[i]]$tot.withinss})
qplot(k, tot_withinss, geom=c("point", "line"), ylab="Total within-cluster sum of squares") + theme_bw()
```

\begin{figure}

{\centering \includegraphics[width=0.5\linewidth]{09-clustering_files/figure-latex/choosingK-1} 

}

\caption{Variance within the clusters. Total within-cluster sum of squares plotted against k.}(\#fig:choosingK)
\end{figure}

*N.B.* we have set ```nstart=50``` so that the algorithm is started 50 times wi

### DBSCAN
Density-based spatial clustering of applications with noise

#### Algorithm


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


