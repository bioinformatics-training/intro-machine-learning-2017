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

classification and non-linear regression

bias and variance

computational load - finding neighbours and storing the entire training set

k-d tree / linear search

system.time k-d tree search vs linear search

library(class)

class::knn

importance of centering a scaling

increase in neighbours - increase in ties



### Measuring distance between objects

**Euclidean distance:**
\begin{equation}
  distance\left(p,q\right)=\sqrt{\sum_{i=1}^{n} (p_i-q_i)^2}
  (\#eq:euclidean)
\end{equation}


<div class="figure" style="text-align: center">
<img src="04-nearest-neighbours_files/figure-html/euclideanDistanceDiagram-1.png" alt="Euclidean distance." width="75%" />
<p class="caption">(\#fig:euclideanDistanceDiagram)Euclidean distance.</p>
</div>

## Classification

### Algorithm
<div class="figure" style="text-align: center">
<img src="images/knn_classification.svg" alt="Illustration of _k_-nn classification. In this example we have two classes: blue squares and red triangles. The green circle represents a test object. If k=3 (solid line circle) the test object is assigned to the red triangle class. If k=5 the test object is assigned to the blue square class.  By Antti Ajanki AnAj - Own work, CC BY-SA 3.0, https://commons.wikimedia.org/w/index.php?curid=2170282" width="75%" />
<p class="caption">(\#fig:knnClassification)Illustration of _k_-nn classification. In this example we have two classes: blue squares and red triangles. The green circle represents a test object. If k=3 (solid line circle) the test object is assigned to the red triangle class. If k=5 the test object is assigned to the blue square class.  By Antti Ajanki AnAj - Own work, CC BY-SA 3.0, https://commons.wikimedia.org/w/index.php?curid=2170282</p>
</div>

### Simulated data
We will use a simulated data set to demonstrate:

* bias-variance trade-off
* the knn function in R
* plotting decision boundaries
* choosing the optimum value of _k_

The dataset is partitioned into training and test sets.

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


### knn function
For _k_-nn classification and regression we will use the **knn** function in the package **class**.

```r
library(class)
```

**Arguments to knn**

* ```train``` : matrix or data frame of training set cases.
* ```test``` : matrix or data frame of test set cases. A vector will be interpreted as a row vector for a single case.
* ```cl``` : factor of true classifications of training set
* ```k``` : number of neighbours considered.
* ```l``` : minimum vote for definite decision, otherwise doubt. (More precisely, less than k-l dissenting votes are allowed, even if k is increased by ties.)
* ```prob``` : If this is true, the proportion of the votes for the winning class are returned as attribute prob.
* ```use.all``` : controls handling of ties. If true, all distances equal to the kth largest are included. If false, a random selection of distances equal to the kth is chosen to use exactly k neighbours.

Perform _k_-nn on the training set with _k_=1

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

```r
cat("KNN prediction error for training set: ", 1-mean(as.numeric(as.vector(knn1train))==ytrain), "\n")
```

```
## KNN prediction error for training set:  0
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

```r
cat("KNN prediction error for test set: ", 1-mean(as.numeric(as.vector(knn1test))==ytest), "\n")
```

```
## KNN prediction error for test set:  0.375
```

### Plotting decision boundaries
Since we have just two dimensions we can visualize the decision boundary generated by the _k_-nn classifier in a 2D scatterplot. Situations where your original data set contains only two variables will be rare, but it is not unusual to reduce a high-dimensional data set to just two dimensions using the methods that will be discussed in chapter \@ref(dimensionality-reduction). Therefore, knowing how to plot decision boundaries will potentially be helpful for many different datasets and classifiers.

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
xgrid <- cbind(xgrid, V3)
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
  geom_contour(data=xgrid, aes(x=V1, y=V2, z=V3), breaks=0.5, col="grey30") +
  ggtitle("train") +
  theme_bw() +
  theme(plot.title = element_text(size=25, face="bold"), axis.text=element_text(size=15),
        axis.title=element_text(size=20,face="bold"))

ggplot(xgrid, aes(V1,V2)) +
  geom_point(col=point_colours[knn1grid], shape=16, size=0.3) +
  geom_point(data=xtest, aes(V1,V2), col=point_colours[ytest+1],
             shape=point_shapes[ytrain+1], size=point_size) +
  geom_contour(data=xgrid, aes(x=V1, y=V2, z=V3), breaks=0.5, col="grey30") +
  ggtitle("test") +
  theme_bw() +
  theme(plot.title = element_text(size=25, face="bold"), axis.text=element_text(size=15),
        axis.title=element_text(size=20,face="bold"))
```

<div class="figure" style="text-align: center">
<img src="04-nearest-neighbours_files/figure-html/simDataBinClassDecisionBoundaryK1-1.png" alt="Binary classification of the simulated training and test sets with _k_=1." width="50%" /><img src="04-nearest-neighbours_files/figure-html/simDataBinClassDecisionBoundaryK1-2.png" alt="Binary classification of the simulated training and test sets with _k_=1." width="50%" />
<p class="caption">(\#fig:simDataBinClassDecisionBoundaryK1)Binary classification of the simulated training and test sets with _k_=1.</p>
</div>

### Bias-variance tradeoff
The bias–variance tradeoff is the problem of simultaneously minimizing two sources of error that prevent supervised learning algorithms from generalizing beyond their training set:

* The bias is error from erroneous assumptions in the learning algorithm. High bias can cause an algorithm to miss the relevant relations between features and target outputs (underfitting).
* The variance is error from sensitivity to small fluctuations in the training set. High variance can cause an algorithm to model the random noise in the training data, rather than the intended outputs (overfitting).


Logarithmic spaced sequence function from [emdbook](https://cran.r-project.org/package=emdbook) package.

```r
lseq <- function(from, to, length.out) {
  exp(seq(log(from), log(to), length.out = length.out))
}
```

Get log spaced sequence of length 20, round and then remove any duplicates resulting from rounding.

```r
s <- unique(round(lseq(1,400,20)))
```


```r
train_error <- sapply(s, function(i){
  yhat <- knn(xtrain, xtrain, ytrain, i)
  return(1-mean(as.numeric(as.vector(yhat))==ytrain))
})

test_error <- sapply(s, function(i){
  yhat <- knn(xtrain, xtest, ytrain, i)
  return(1-mean(as.numeric(as.vector(yhat))==ytest))
})

k <- rep(s, 2)
set <- c(rep("train", length(s)), rep("test", length(s)))
error <- c(train_error, test_error)
misclass_errors <- data.frame(k, set, error)
```


```r
ggplot(misclass_errors, aes(x=k, y=error, group=set)) + 
  geom_line(aes(colour=set, linetype=set), size=1.5) +
  scale_x_log10() +
  ylab("Misclassification Errors") +
  theme_bw() +
  theme(legend.position = c(0.5, 0.25), legend.title=element_blank(),
        legend.text=element_text(size=12), 
        axis.title.x=element_text(face="italic", size=12))
```

<div class="figure" style="text-align: center">
<img src="04-nearest-neighbours_files/figure-html/misclassErrorsFunK-1.png" alt="Misclassification errors as a function of neighbourhood size." width="100%" />
<p class="caption">(\#fig:misclassErrorsFunK)Misclassification errors as a function of neighbourhood size.</p>
</div>

### Choosing _k_

We will use the caret library.

```r
library(caret)
```

```
## Loading required package: lattice
```

[caret](http://cran.r-project.org/web/packages/caret/index.html) has automatic parallel processing built in. To take advantage of this feature we simply need to load the [doMC](http://cran.r-project.org/web/packages/doMC/index.html) package and register workers: 

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

The [caret](http://cran.r-project.org/web/packages/caret/index.html) function **train** is used to fit predictive models over different values of _k_. The function **trainControl** is used to specify a list of computational and resampling options, which will be passed to **train**. We will start by configuring our cross-validation procedure using **trainControl**.

We would like to make this demonstration reproducible and because we will be running the models in parallel, using the **set.seed** function alone is not sufficient. In addition to using **set.seed** we have to make use of the optional **seeds** argument to **trainControl**. We need to supply **seeds** with a list of integers that will be used to set the seed at each sampling iteration. The list is required to have a length of B+1, where B is the number of resamples. We will be repeating 10-fold cross-validation a total of ten times and so our list must have a length of 101. The first B elements of the list are required to be vectors of integers of length M, where M is the number of models being evaluated (in this case 19). The last element of the list only needs to be a single integer, which will be used for the final model.

First we generate our list of seeds.

```r
set.seed(42)
seeds <- vector(mode = "list", length = 101)
for(i in 1:100) seeds[[i]] <- sample.int(1000, 19)
seeds[[101]] <- sample.int(1000,1)
```

We can now use **trainControl** to create a list of computational options for resampling.

```r
tc <- trainControl(method="repeatedcv",
                   number = 10,
                   repeats = 10,
                   seeds = seeds)
```

There are two options for choosing the values of _k_ to be evaluated by the **train** function:
1. Pass a data.frame of values of _k_ to the **tuneGrid** argument of **train**.
2. Specify the number of different levels of _k_ using the **tuneLength** function and allow **train** to pick the actual values.

We will use the first option, so that we can try the values of _k_ we examined earlier. We need to convert the vector of values of k we created earlier and convert it into a data.frame.


```r
s <- data.frame(s)
names(s) <- "k"
```

We are now ready to run the cross-validation.

```r
knnFit <- train(xtrain, as.factor(ytrain), 
                method="knn",
                tuneGrid=s,
                trControl=tc)

knnFit
```

```
## k-Nearest Neighbors 
## 
## 400 samples
##   2 predictor
##   2 classes: '0', '1' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold, repeated 10 times) 
## Summary of sample sizes: 360, 360, 360, 360, 360, 360, ... 
## Resampling results across tuning parameters:
## 
##   k    Accuracy  Kappa 
##     1  0.63375   0.2675
##     2  0.64125   0.2825
##     3  0.67925   0.3585
##     4  0.67200   0.3440
##     5  0.69675   0.3935
##     7  0.71100   0.4220
##     9  0.71650   0.4330
##    12  0.71450   0.4290
##    17  0.72650   0.4530
##    23  0.73175   0.4635
##    32  0.73775   0.4755
##    44  0.74075   0.4815
##    60  0.74675   0.4935
##    83  0.75475   0.5095
##   113  0.73600   0.4720
##   155  0.72500   0.4500
##   213  0.70950   0.4190
##   292  0.69300   0.3860
##   400  0.51300   0.0260
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was k = 83.
```

**Cohen's Kappa:**
\begin{equation}
  Kappa = \frac{O-E}{1-E}
  (\#eq:kappa)
\end{equation}

where _O_ is the observed accuracy and _E_ is the expected accuracy based on the marginal totals of the confusion matrix. Cohen's Kappa takes values between -1 and 1; a value of zero indicates no agreement between the observed and predicted classes, while a value of one shows perfect concordance of the model prediction and the observed classes. If the prediction is in the opposite direction of the truth, a negative value will be obtained, but large negative values are rare in practice [@Kuhn2013].

We can plot accuracy (determined from repeated cross-validation) as a function of neighbourhood size.

```r
plot(knnFit)
```

<div class="figure" style="text-align: center">
<img src="04-nearest-neighbours_files/figure-html/cvAccuracyFunK-1.png" alt="Accuracy (repeated cross-validation) as a function of neighbourhood size." width="100%" />
<p class="caption">(\#fig:cvAccuracyFunK)Accuracy (repeated cross-validation) as a function of neighbourhood size.</p>
</div>

### Feature selection







Error
training vs cv vs test

## Regression
<!--
## Caret

pre-processing
identification of correlated predictors


Parallel processing with doMC
registerDoMC()
getDoParWorkers()

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

-->


## Exercises

### Exercise 1 {#knnEx1}
Classification

Try different methods of feature selection

### Exercise 2 {#knnEx2}
Regression

Alzheimers & gene expression? MMSE and gene expression?



Solutions to exercises can be found in appendix \@ref(solutions-nearest-neighbours).
