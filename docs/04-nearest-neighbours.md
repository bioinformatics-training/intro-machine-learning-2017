# Nearest neighbours {#nearest-neighbours}

<!-- Matt -->

<!-- 
Get ideas on presentation from Harvard bioinformatics website. In particular, use of dataset with two variables (crabs??), because easier to display. Performance of classifier as k increases (should initially improve and then get worse - starts to lose flexibility).

In exercises could introduce application of knn to regression.

GENERAL:
SPLOM for displaying datasets with small number of variables

FEATURE SELECTION
filter methods  /  wrapper methods / genetic algorithms

Refer to scikit learn

FEATURE SCALING

BIAS-VARIANCE TRADEOFF
In statistics and machine learning, the bias–variance tradeoff (or dilemma) is the problem of simultaneously minimizing two sources of error that prevent supervised learning algorithms from generalizing beyond their training set[citation needed].:

    The bias is error from erroneous assumptions in the learning algorithm. High bias can cause an algorithm to miss the relevant relations between features and target outputs (underfitting).
    The variance is error from sensitivity to small fluctuations in the training set. High variance can cause an algorithm to model the random noise in the training data, rather than the intended outputs (overfitting).


-->
## Introduction
memory based and require no model to be fit

bias and variance

computational load - finding neighbours and storing the entire training set

k-d tree / linear search

system.time k-d tree search vs linear search

library(class)

class::knn

importance of centering a scaling

increase in neighbours - increase in ties

## Algorithm

### Defining nearest

**Euclidean distance:**
\begin{equation}
  distance\left(p,q\right)=\sqrt{\sum_{i=1}^{n} (p_i-q_i)^2}
  (\#eq:euclidean)
\end{equation}


<div class="figure" style="text-align: center">
<img src="04-nearest-neighbours_files/figure-html/euclideanDistanceDiagram-1.png" alt="Euclidean distance." width="75%" />
<p class="caption">(\#fig:euclideanDistanceDiagram)Euclidean distance.</p>
</div>


<div class="figure" style="text-align: center">
<img src="images/knn_classification.svg" alt="Illustration of _k_-nn classification. In this example we have two classes: blue squares and red triangles. The green circle represents a test object. If k=3 (solid line circle) the test object is assigned to the red triangle class. If k=5 the test object is assigned to the blue square class.  By Antti Ajanki AnAj - Own work, CC BY-SA 3.0, https://commons.wikimedia.org/w/index.php?curid=2170282" width="75%" />
<p class="caption">(\#fig:knnClassification)Illustration of _k_-nn classification. In this example we have two classes: blue squares and red triangles. The green circle represents a test object. If k=3 (solid line circle) the test object is assigned to the red triangle class. If k=5 the test object is assigned to the blue square class.  By Antti Ajanki AnAj - Own work, CC BY-SA 3.0, https://commons.wikimedia.org/w/index.php?curid=2170282</p>
</div>





## Classification
Error
training vs cv vs test

## Regression

## Caret

pre-processing
identification of correlated predictors


Parallel processing with doMC
registerDoMC()
getDoParWorkers()

## Example one

Load data

```r
load("data/example_binary_classification/bin_class_example.rda")
str(xtrain)
```

```
## 'data.frame':	400 obs. of  2 variables:
##  $ V1: num  -0.223 0.944 2.36 1.846 1.732 ...
##  $ V2: num  -1.153 -0.827 -0.128 2.014 -0.574 ...
```

```r
str(xtest)
```

```
## 'data.frame':	400 obs. of  2 variables:
##  $ V1: num  2.09 2.3 2.07 1.65 1.18 ...
##  $ V2: num  -1.009 1.0947 0.1644 0.3243 -0.0277 ...
```

```r
summary(as.factor(ytrain))
```

```
##   0   1 
## 200 200
```

```r
summary(as.factor(ytest))
```

```
##   0   1 
## 200 200
```


```r
library(ggplot2)
library(GGally)
library(RColorBrewer)
point_shapes <- c(15,17)
point_colours <- brewer.pal(3,"Dark2")
point_size = 2

ggplot(xtrain, aes(V1,V2)) + 
  geom_point(col=point_colours[ytrain+1], shape=point_shapes[ytrain+1], 
             size=point_size) + 
  ggtitle("train") +
  theme_bw() +
  theme(plot.title = element_text(size=25, face="bold"), axis.text=element_text(size=15),
        axis.title=element_text(size=20,face="bold"))

ggplot(xtest, aes(V1,V2)) + 
  geom_point(col=point_colours[ytest+1], shape=point_shapes[ytest+1], 
             size=point_size) + 
  ggtitle("test") +
  theme_bw() +
  theme(plot.title = element_text(size=25, face="bold"), axis.text=element_text(size=15),
        axis.title=element_text(size=20,face="bold"))
```

<div class="figure" style="text-align: center">
<img src="04-nearest-neighbours_files/figure-html/simDataBinClassTrainTest-1.png" alt="Scatterplots of the simulated training and test data sets that will be used in the demonstration of binary classification using _k_-nn" width="50%" /><img src="04-nearest-neighbours_files/figure-html/simDataBinClassTrainTest-2.png" alt="Scatterplots of the simulated training and test data sets that will be used in the demonstration of binary classification using _k_-nn" width="50%" />
<p class="caption">(\#fig:simDataBinClassTrainTest)Scatterplots of the simulated training and test data sets that will be used in the demonstration of binary classification using _k_-nn</p>
</div>

For _k_-nn classification and regression we will use the **knn** function in the package **class**.

```r
library(class)
```

**Arguments to knn**

```train``` : matrix or data frame of training set cases.

```test``` : matrix or data frame of test set cases. A vector will be interpreted as a row vector for a single case.

```cl``` : factor of true classifications of training set

```k``` : number of neighbours considered.

```l``` : minimum vote for definite decision, otherwise doubt. (More precisely, less than k-l dissenting votes are allowed, even if k is increased by ties.)

```prob``` : If this is true, the proportion of the votes for the winning class are returned as attribute prob.

```use.all``` : controls handling of ties. If true, all distances equal to the kth largest are included. If false, a random selection of distances equal to the kth is chosen to use exactly k neighbours.

Training data set

```r
knn1train <- class::knn(train=xtrain, test=xtrain, cl=ytrain, k=1)
table(ytrain,knn1train)
```

```
##       knn1train
## ytrain   0   1
##      0 200   0
##      1   0 200
```

Test data set

```r
knn1test <- class::knn(train=xtrain, test=xtest, cl=ytrain, k=1)
table(ytest, knn1test)
```

```
##      knn1test
## ytest   0   1
##     0 131  69
##     1  81 119
```

When we have just two dimensions it is easy to visualize the decision boundary generated by the _k_-nn classifier. Obviously quite unusual to be dealing with just two variables, but we may have reduced a high dimensional dataset to just two dimensions using the techniques that will be discussed in chapter \@ref(dimensionality-reduction).

Create a grid so we can predict across the full range of our variables V1 and V2.

```r
gridSize <- 150 
v1limits <- c(min(c(xtrain[,1],xtest[,1])),max(c(xtrain[,1],xtest[,1])))
tmpV1 <- seq(v1limits[1],v1limits[2],len=gridSize)
v2limits <- c(min(c(xtrain[,2],xtest[,2])),max(c(xtrain[,2],xtest[,2])))
tmpV2 <- seq(v2limits[1],v2limits[2],len=gridSize)
xgrid <- expand.grid(tmpV1,tmpV2)
names(xgrid) <- names(xtrain)
```

Predict values of all elements of grid.

```r
knn1grid <- class::knn(train=xtrain, test=xgrid, cl=ytrain, k=1)
V3 <- as.numeric(as.vector(knn1grid))
xgrid2 <- cbind(xgrid, V3)
```

Plot

```r
point_shapes <- c(15,17)
point_colours <- brewer.pal(3,"Dark2")
point_size = 2
# grid point 16
# grid point size =0.2

ggplot(xgrid, aes(V1,V2)) +
  geom_point(col=point_colours[knn1grid], shape=16, size=0.3) +
  geom_point(data=xtrain, aes(V1,V2), col=point_colours[ytrain+1],
             shape=point_shapes[ytrain+1], size=point_size) +
  geom_contour(data=xgrid2, aes(x=V1, y=V2, z=V3), breaks=c(0,.5), col="grey30") +
  ggtitle("train") +
  theme_bw() +
  theme(plot.title = element_text(size=25, face="bold"), axis.text=element_text(size=15),
        axis.title=element_text(size=20,face="bold"))

ggplot(xgrid, aes(V1,V2)) +
  geom_point(col=point_colours[knn1grid], shape=16, size=0.3) +
  geom_point(data=xtest, aes(V1,V2), col=point_colours[ytest+1],
             shape=point_shapes[ytrain+1], size=point_size) +
  geom_contour(data=xgrid2, aes(x=V1, y=V2, z=V3), breaks=c(0,.5), col="grey30") +
  ggtitle("test") +
  theme_bw() +
  theme(plot.title = element_text(size=25, face="bold"), axis.text=element_text(size=15),
        axis.title=element_text(size=20,face="bold"))
```

<div class="figure" style="text-align: center">
<img src="04-nearest-neighbours_files/figure-html/simDataBinClassDecisionBoundaryK1-1.png" alt="Binary classification of the simulated training and test sets with _k_=1." width="50%" /><img src="04-nearest-neighbours_files/figure-html/simDataBinClassDecisionBoundaryK1-2.png" alt="Binary classification of the simulated training and test sets with _k_=1." width="50%" />
<p class="caption">(\#fig:simDataBinClassDecisionBoundaryK1)Binary classification of the simulated training and test sets with _k_=1.</p>
</div>


## Example two

## Curse of dimensionality
Pre-processing data using dimensionality reduction.

transformation functionality in caret

## Examples

centre1 <- read.csv("data/serum_proteomics/male_centre1.csv")
centre2 <- read.csv("data/serum_proteomics/male_centre2.csv")

c1sub <- centre1[,c(1,5,6,9,10)]
c2sub <- centre2[,c(1,5,6,9,10)]

res <- FNN::knn(c1sub[,2:5], c1sub[,2:5], cl=c1sub$Diagnostic_group, k=1)
table(c1sub$Diagnostic_group, res)

res <- FNN::knn(c1sub[,2:5], c2sub[,2:5], cl=c1sub$Diagnostic_group, k=1)
table(c2sub$Diagnostic_group, res)

bias / variance trade-off

include:
division into training and test set
preprocessing - illustrate with diagram




## Exercises

### Exercise 1 {#knnEx1}
Classification

Try different methods of feature selection

### Exercise 2 {#knnEx2}
Regression

Alzheimers & gene expression? MMSE and gene expression?



Solutions to exercises can be found in appendix \@ref(solutions-nearest-neighbours).
