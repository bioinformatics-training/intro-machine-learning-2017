# Support vector machines {#svm}

## Introduction
Support vector machines (SVMs) are models of supervised learning. The SVM approach emerged from the computer science community in the 1990s. This chapter will focus exclusively on the use of SVMs for classification, but it should be noted that they can also be used for regression. The SVM is an extension of the support vector classifier (SVC), which is turn is an extension of the maximum margin classifier.

### Maximum margin classifier
Let's start by definining a hyperplane. In _p_-dimensional space a hyperplane is a flat affine subspace of _p_-1. 

<div class="figure" style="text-align: center">
<img src="images/Svm_separating_hyperplanes.svg" alt="Separating hyperplanes. H1 does not separate the classes. H2 does, but only with a small margin. H3 separates them with the maximum margin. By User:ZackWeinberg, based on PNG version by User:Cyc [CC BY-SA 3.0 (https://creativecommons.org/licenses/by-sa/3.0)], via Wikimedia Commons" width="75%" />
<p class="caption">(\#fig:svmSeparatingHyperplanes)Separating hyperplanes. H1 does not separate the classes. H2 does, but only with a small margin. H3 separates them with the maximum margin. By User:ZackWeinberg, based on PNG version by User:Cyc [CC BY-SA 3.0 (https://creativecommons.org/licenses/by-sa/3.0)], via Wikimedia Commons</p>
</div>


<div class="figure" style="text-align: center">
<img src="images/Svm_max_sep_hyperplane_with_margin.png" alt="Maximum-margin hyperplane and margins for an SVM trained with samples from two classes. Samples on the margin are called the support vectors. By Cyc - Own work, Public Domain, https://commons.wikimedia.org/w/index.php?curid=3566688" width="75%" />
<p class="caption">(\#fig:svmMaxSepHyperplaneWithMargin)Maximum-margin hyperplane and margins for an SVM trained with samples from two classes. Samples on the margin are called the support vectors. By Cyc - Own work, Public Domain, https://commons.wikimedia.org/w/index.php?curid=3566688</p>
</div>


<!-- Matt -->

<!--REPEAT CLASSIFICATION EXAMPLE FOR SERUM PROTEOMICS -->

<!-- regression and classification -->

## Support vector classifier

<!--First generate a data set.-->
<div class="figure" style="text-align: center">
<img src="06-svm_files/figure-html/svcTestData-1.png" alt="Example data with linearly separable groups." width="75%" />
<p class="caption">(\#fig:svcTestData)Example data with linearly separable groups.</p>
</div>

<div class="figure" style="text-align: center">
<img src="06-svm_files/figure-html/svcCost10-1.png" alt="Support vector classifier with cost=10." width="75%" />
<p class="caption">(\#fig:svcCost10)Support vector classifier with cost=10.</p>
</div>

<div class="figure" style="text-align: center">
<img src="06-svm_files/figure-html/svcCost01-1.png" alt="Support vector classifier with cost=0.1." width="75%" />
<p class="caption">(\#fig:svcCost01)Support vector classifier with cost=0.1.</p>
</div>

## Support Vector Machine
<div class="figure" style="text-align: center">
<img src="06-svm_files/figure-html/svmExampleData-1.png" alt="Example data for demonstrating SVM." width="75%" />
<p class="caption">(\#fig:svmExampleData)Example data for demonstrating SVM.</p>
</div>

<div class="figure" style="text-align: center">
<img src="images/svm_kernel_machine.png" alt="Kernel machine. By Alisneaky - Own work, CC0, https://commons.wikimedia.org/w/index.php?curid=14941564" width="75%" />
<p class="caption">(\#fig:svmKernelMachine)Kernel machine. By Alisneaky - Own work, CC0, https://commons.wikimedia.org/w/index.php?curid=14941564</p>
</div>

<div class="figure" style="text-align: center">
<img src="06-svm_files/figure-html/svmExampleCost1-1.png" alt="SVM with cost 1." width="75%" />
<p class="caption">(\#fig:svmExampleCost1)SVM with cost 1.</p>
</div>

<div class="figure" style="text-align: center">
<img src="06-svm_files/figure-html/svmExampleCost1e5-1.png" alt="SVM with cost 100000." width="75%" />
<p class="caption">(\#fig:svmExampleCost1e5)SVM with cost 100000.</p>
</div>

## Example - training an SVM
Training of an SVM will be demonstrated on a 2-dimensional simulated data set, with a non-linear decision boundary.

### Setup environment
Load required libraries

```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(RColorBrewer)
library(ggplot2)
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
library(pROC)
```

```
## Type 'citation("pROC")' for a citation.
```

```
## 
## Attaching package: 'pROC'
```

```
## The following objects are masked from 'package:stats':
## 
##     cov, smooth, var
```

Initialize parallel processing

```r
registerDoMC()
getDoParWorkers()
```

```
## [1] 2
```

### Partition data
Load data

```r
moons <- read.csv("data/sim_data_svm/moons.csv", header=F)
str(moons)
```

```
## 'data.frame':	400 obs. of  3 variables:
##  $ V1: num  -0.496 1.827 1.322 -1.138 -0.21 ...
##  $ V2: num  0.985 -0.501 -0.397 0.192 -0.145 ...
##  $ V3: Factor w/ 2 levels "A","B": 1 2 2 1 2 1 1 2 1 2 ...
```

V1 and V2 are the predictors; V3 is the class. 

Partition data into training and test set

```r
set.seed(42)
trainIndex <- createDataPartition(y=moons$V3, times=1, p=0.7, list=F)
moonsTrain <- moons[trainIndex,]
moonsTest <- moons[-trainIndex,]

summary(moonsTrain$V3)
```

```
##   A   B 
## 140 140
```

```r
summary(moonsTest$V3)
```

```
##  A  B 
## 60 60
```

### Visualize training data

```r
point_shapes <- c(15,17)
bp <- brewer.pal(3,"Dark2")
point_colours <- ifelse(moonsTrain$V3=="A", bp[1], bp[2])
point_shapes <- ifelse(moonsTrain$V3=="A", 15, 17)

point_size = 2

ggplot(moonsTrain, aes(V1,V2)) + 
  geom_point(col=point_colours, shape=point_shapes, 
             size=point_size) + 
  theme_bw() +
  theme(plot.title = element_text(size=25, face="bold"), axis.text=element_text(size=15),
        axis.title=element_text(size=20,face="bold"))
```

<div class="figure" style="text-align: center">
<img src="06-svm_files/figure-html/svmMoonsTrainSet-1.png" alt="Scatterplot of the training data" width="50%" />
<p class="caption">(\#fig:svmMoonsTrainSet)Scatterplot of the training data</p>
</div>


### Model cross-validation and tuning
Set seeds for reproducibility. We will be trying 9 values of the tuning parameter with 10 repeats of 10 fold cross-validation, so we need the following list of seeds.

```r
set.seed(42)
seeds <- vector(mode = "list", length = 101)
for(i in 1:100) seeds[[i]] <- sample.int(1000, 9)
seeds[[101]] <- sample.int(1000,1)
```

We will pass the twoClassSummary function into model training through trainControl. Additionally we would like the model to predict class probabilities so that we can calculate the ROC curve, so we use the classProbs option.

```r
cvCtrl <- trainControl(method = "repeatedcv", 
                       repeats = 5,
                       number = 5,
                       summaryFunction = twoClassSummary,
                       classProbs = TRUE,
                       seeds=seeds)
```

We set the **method** of the **train** function to **svmRadial** to specify a radial kernel SVM. In this implementation we only have to tune one parameter, **cost**. An appropriate value of the **sigma** parameter (used to the kernel feature space) is estimated from the data. The default grid of cost parameters start at 0.25 and double at each iteration. Choosing tuneLength = 9 will give us cost parameters of 0.25, 0.5, 1, 2, 4, 8, 16, 32 and 64. 

```r
svmTune <- train(x = moonsTrain[,c(1:2)],
                 y = moonsTrain[,3],
                 method = "svmRadial",
                 tuneLength = 9,
                 preProc = c("center", "scale"),
                 metric = "ROC",
                 trControl = cvCtrl)
```

```
## Loading required package: kernlab
```

```
## 
## Attaching package: 'kernlab'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     alpha
```

```r
svmTune
```

```
## Support Vector Machines with Radial Basis Function Kernel 
## 
## 280 samples
##   2 predictor
##   2 classes: 'A', 'B' 
## 
## Pre-processing: centered (2), scaled (2) 
## Resampling: Cross-Validated (5 fold, repeated 5 times) 
## Summary of sample sizes: 224, 224, 224, 224, 224, 224, ... 
## Resampling results across tuning parameters:
## 
##   C      ROC        Sens       Spec     
##    0.25  0.9531122  0.8900000  0.8957143
##    0.50  0.9573980  0.8857143  0.9071429
##    1.00  0.9588265  0.8828571  0.9042857
##    2.00  0.9570918  0.8800000  0.8914286
##    4.00  0.9583673  0.8714286  0.8942857
##    8.00  0.9602041  0.8657143  0.8857143
##   16.00  0.9582143  0.8671429  0.8857143
##   32.00  0.9546939  0.8571429  0.8971429
##   64.00  0.9488265  0.8500000  0.8871429
## 
## Tuning parameter 'sigma' was held constant at a value of 1.100359
## ROC was used to select the optimal model using  the largest value.
## The final values used for the model were sigma = 1.100359 and C = 8.
```


```r
svmTune$finalModel
```

```
## Support Vector Machine object of class "ksvm" 
## 
## SV type: C-svc  (classification) 
##  parameter : cost C = 8 
## 
## Gaussian Radial Basis kernel function. 
##  Hyperparameter : sigma =  1.10035896819859 
## 
## Number of Support Vectors : 83 
## 
## Objective Function Value : -478.4363 
## Training error : 0.110714 
## Probability model included.
```

### Prediction performance measures
SVM accuracy profile

```r
plot(svmTune, metric = "ROC", scales = list(x = list(log =2)))
```

<div class="figure" style="text-align: center">
<img src="06-svm_files/figure-html/svmAccuracyProfileMoons-1.png" alt="SVM accuracy profile for moons data set." width="80%" />
<p class="caption">(\#fig:svmAccuracyProfileMoons)SVM accuracy profile for moons data set.</p>
</div>

Predictions on test set.

```r
svmPred <- predict(svmTune, moonsTest[,c(1:2)])
confusionMatrix(svmPred, moonsTest[,3])
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction  A  B
##          A 55  5
##          B  5 55
##                                           
##                Accuracy : 0.9167          
##                  95% CI : (0.8521, 0.9593)
##     No Information Rate : 0.5             
##     P-Value [Acc > NIR] : <2e-16          
##                                           
##                   Kappa : 0.8333          
##  Mcnemar's Test P-Value : 1               
##                                           
##             Sensitivity : 0.9167          
##             Specificity : 0.9167          
##          Pos Pred Value : 0.9167          
##          Neg Pred Value : 0.9167          
##              Prevalence : 0.5000          
##          Detection Rate : 0.4583          
##    Detection Prevalence : 0.5000          
##       Balanced Accuracy : 0.9167          
##                                           
##        'Positive' Class : A               
## 
```

Get predicted class probabilities so we can build ROC curve.

```r
svmProbs <- predict(svmTune, moonsTest[,c(1:2)], type="prob")
head(svmProbs)
```

```
##            A           B
## 1 0.06833065 0.931669353
## 2 0.07460720 0.925392800
## 3 0.99189246 0.008107535
## 4 0.98795124 0.012048763
## 5 0.05141950 0.948580502
## 6 0.92523623 0.074763767
```

Build a ROC curve.

```r
svmROC <- roc(moonsTest[,3], svmProbs[,"A"])
auc(svmROC)
```

```
## Area under the curve: 0.9575
```

Plot ROC curve, including the threshold with the highest sum sensitivity + specificity.

```r
plot(svmROC, type = "S", 
     print.thres = "best",
     print.thres.col = "blue",
     print.thres.pch = 19,
     print.thres.cex=1.5)
```

<div class="figure" style="text-align: center">
<img src="06-svm_files/figure-html/svmROCcurveMoons-1.png" alt="SVM accuracy profile." width="80%" />
<p class="caption">(\#fig:svmROCcurveMoons)SVM accuracy profile.</p>
</div>

Calculate area under ROC curve. 

```r
auc(svmROC)
```

```
## Area under the curve: 0.9575
```


## Exercises

### Exercise 1

In this exercise we will return to the cell segmentation data set that we attempted to classify in section \@knn-cell-segmentation of the nearest neighbours chapter.

```r
data(segmentationData)
```

The aim of the exercise is to build a binary classifier to predict the quality of segmentation (poorly segmented or well segmented) based on the various morphological features. Do not worry about feature selection, but you may want to pre-process the data. Select a radial kernel SVM and tune over the cost function C. Produce a ROC curve to show the performance of the classifier on the test set. 


Solutions to exercises can be found in appendix \@ref(solutions-svm)
