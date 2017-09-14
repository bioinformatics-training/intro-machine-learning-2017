# Solutions ch. 10 - Clustering {#solutions-clustering}

Solutions to exercises of chapter \@ref(clustering). 

<!--
set.seed(42)
dbscanRes <- dbscan::dbscan(t(e), eps=85, minPts=5)


d <- dist((t(e)[dbscanRes$cluster>0,]))

dbscanS <- silhouette(dbscanRes$cluster[dbscanRes$cluster>0], d)
plot(dbscanS, col="black")

-->


```r
library(ggplot2)
```


## Exercise 1


## Exercise 2

First we need to read the image data and transform it into a suitable format for analysis:




```r
library(EBImage)
img <- readImage("data/histology/Emphysema_H_and_E.jpg")

imgDim <- dim(img)
```


```r
imgDF <- data.frame(
  x = rep(1:imgDim[1], imgDim[2]),
  y = rep(imgDim[2]:1, each=imgDim[1]),
  r = as.vector(img[,,1]),
  g = as.vector(img[,,2]),
  b = as.vector(img[,,3])
)
```

Next we will perform kmeans clustering for k in the range 1:9. This is computationally quite intensive, so we'll use parallel processing:

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
k=1:9
set.seed(42)
res_k_9 <- foreach(
  i=k, 
  .options.multicore=list(set.seed=FALSE)) %dopar% kmeans(imgDF[,c("r", "g", "b")], i, nstart=50)
```

We can now plot total within-cluster sum of squares against k:

```r
plot_tot_withinss <- function(kmeans_output){
  tot_withinss <- sapply(k, function(i){kmeans_output[[i]]$tot.withinss})
  qplot(k, tot_withinss, geom=c("point", "line"), 
        ylab="Total within-cluster sum of squares") + theme_bw()
}

plot_tot_withinss(res_k_9)
```

<div class="figure" style="text-align: center">
<img src="20-solutions-clustering_files/figure-html/imageAnalysisElbowPlot-1.png" alt="Variance within the clusters of pixels. Total within-cluster sum of squares plotted against k." width="80%" />
<p class="caption">(\#fig:imageAnalysisElbowPlot)Variance within the clusters of pixels. Total within-cluster sum of squares plotted against k.</p>
</div>




```r
clusterColours <- rgb(res_k_9[[4]]$centers) 
ggplot(data = imgDF, aes(x = x, y = y)) + 
  geom_point(colour = clusterColours[res_k_9[[4]]$cluster]) +
  xlab("x") +
  ylab("y") +
  theme_minimal()
```

<img src="20-solutions-clustering_files/figure-html/unnamed-chunk-6-1.png" width="672" />



```r
clusterColours <- rgb(res_k_9[[2]]$centers) 
ggplot(data = imgDF, aes(x = x, y = y)) + 
  geom_point(colour = clusterColours[res_k_9[[2]]$cluster]) +
  xlab("x") +
  ylab("y") +
  theme_minimal()
```

<img src="20-solutions-clustering_files/figure-html/unnamed-chunk-7-1.png" width="672" />

