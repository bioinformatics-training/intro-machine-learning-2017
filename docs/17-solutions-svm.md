# Solutions ch. 7 - Support vector machines {#solutions-svm}

Solutions to exercises of chapter \@ref(svm).

## Exercise 1

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

Setup parallel processing

```r
registerDoMC()
getDoParWorkers()
```

```
## [1] 2
```

MUST GENERATE A LIST OF SEEDS IF WE USE PARALLEL PROCESSING, FOR REPRODUCIBILITY

Load data

```r
data(segmentationData)
```


```r
segClass <- segmentationData$Class
```

Extract predictors from segmentationData

```r
segData <- segmentationData[,4:61]
```

Partition data

```r
set.seed(42)
trainIndex <- createDataPartition(y=segClass, times=1, p=0.5, list=F)
segDataTrain <- segData[trainIndex,]
segDataTest <- segData[-trainIndex,]
segClassTrain <- segClass[trainIndex]
segClassTest <- segClass[-trainIndex]
```

We already know what pre-processing steps are required for this data set, having worked with it before in section \@knn-cell-segmentation of the nearest neighbours chapter.

```r
transformations <- preProcess(segDataTrain, 
                              method=c("YeoJohnson", "center", "scale", "corr"),
                              cutoff=0.75)
segDataTrain <- predict(transformations, segDataTrain)
```

Set seeds for reproducibility (optional). We will be trying 9 values of the tuning parameter with 5 repeats of 5 fold cross-validation, so we need the following list of seeds.

```r
set.seed(42)
seeds <- vector(mode = "list", length = 26)
for(i in 1:25) seeds[[i]] <- sample.int(1000, 9)
seeds[[26]] <- sample.int(1000,1)
```

We will pass the twoClassSummary function into model training through **trainControl**. Additionally we would like the model to predict class probabilities so that we can calculate the ROC curve, so we use the **classProbs** option. 

```r
cvCtrl <- trainControl(method = "repeatedcv", 
                       repeats = 5,
                       number = 5,
                       summaryFunction = twoClassSummary,
                       classProbs = TRUE,
                       seeds=seeds)
```

Tune SVM over the cost parameter. The default grid of cost parameters start at 0.25 and double at each iteration. Choosing ```tuneLength = 9``` will give us cost parameters of 0.25, 0.5, 1, 2, 4, 8, 16, 32 and 64. The train function will calculate an appropriate value of sigma (the kernel parameter) from the data.

```r
svmTune <- train(x = segDataTrain,
                 y = segClassTrain,
                 method = "svmRadial",
                 tuneLength = 9,
                 metric = "ROC",
                 trControl = cvCtrl)
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

```
## maximum number of iterations reached 1.031922e-05 7.460141e-06
```

```r
svmTune
```

```
## Support Vector Machines with Radial Basis Function Kernel 
## 
## 1010 samples
##   27 predictor
##    2 classes: 'PS', 'WS' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold, repeated 5 times) 
## Summary of sample sizes: 808, 808, 808, 808, 808, 808, ... 
## Resampling results across tuning parameters:
## 
##   C      ROC        Sens       Spec       
##    0.25  0.6966090  0.9984615  0.000000000
##    0.50  0.7503013  0.9984615  0.004444444
##    1.00  0.7026175  0.9981538  0.004444444
##    2.00  0.6329594  0.9978462  0.003333333
##    4.00  0.7502799  0.9978462  0.005000000
##    8.00  0.7254850  0.9981538  0.004444444
##   16.00  0.7292115  0.9975385  0.003888889
##   32.00  0.7502799  0.9981538  0.003888889
##   64.00  0.7230833  0.9981538  0.005000000
## 
## Tuning parameter 'sigma' was held constant at a value of 0.02416426
## ROC was used to select the optimal model using  the largest value.
## The final values used for the model were sigma = 0.02416426 and C = 0.5.
```


```r
svmTune$finalModel
```

```
## Support Vector Machine object of class "ksvm" 
## 
## SV type: C-svc  (classification) 
##  parameter : cost C = 0.5 
## 
## Gaussian Radial Basis kernel function. 
##  Hyperparameter : sigma =  1 
## 
## Number of Support Vectors : 1010 
## 
## Objective Function Value : -290.0686 
## Training error : 0.356436 
## Probability model included.
```

SVM accuracy profile

```r
plot(svmTune, metric = "ROC", scales = list(x = list(log =2)))
```

<div class="figure" style="text-align: center">
<img src="17-solutions-svm_files/figure-html/svmAccuracyProfileCellSegment-1.png" alt="SVM accuracy profile." width="80%" />
<p class="caption">(\#fig:svmAccuracyProfileCellSegment)SVM accuracy profile.</p>
</div>

Test set results

```r
segDataTest <- predict(transformations, segDataTest)
svmPred <- predict(svmTune, segDataTest)
confusionMatrix(svmPred, segClassTest)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction  PS  WS
##         PS 650 357
##         WS   0   2
##                                           
##                Accuracy : 0.6462          
##                  95% CI : (0.6158, 0.6757)
##     No Information Rate : 0.6442          
##     P-Value [Acc > NIR] : 0.462           
##                                           
##                   Kappa : 0.0072          
##  Mcnemar's Test P-Value : <2e-16          
##                                           
##             Sensitivity : 1.000000        
##             Specificity : 0.005571        
##          Pos Pred Value : 0.645482        
##          Neg Pred Value : 1.000000        
##              Prevalence : 0.644202        
##          Detection Rate : 0.644202        
##    Detection Prevalence : 0.998018        
##       Balanced Accuracy : 0.502786        
##                                           
##        'Positive' Class : PS              
## 
```

Get predicted class probabilities

```r
svmProbs <- predict(svmTune, segDataTest, type="prob")
head(svmProbs)
```

```
##          PS        WS
## 1 0.6435817 0.3564183
## 2 0.6436550 0.3563450
## 3 0.6433087 0.3566913
## 4 0.6436212 0.3563788
## 5 0.6436427 0.3563573
## 6 0.6436126 0.3563874
```

Build a ROC curve

```r
svmROC <- roc(segClassTest, svmProbs[,"PS"])
auc(svmROC)
```

```
## Area under the curve: 0.743
```

Plot ROC curve.

```r
plot(svmROC, type = "S", 
     print.thres = 0.5,
     print.thres.col = "blue",
     print.thres.pch = 19,
     print.thres.cex=1.5)
```

<div class="figure" style="text-align: center">
<img src="17-solutions-svm_files/figure-html/svmROCcurveCellSegment-1.png" alt="SVM ROC curve for cell segmentation data set." width="80%" />
<p class="caption">(\#fig:svmROCcurveCellSegment)SVM ROC curve for cell segmentation data set.</p>
</div>

Calculate area under ROC curve

```r
auc(svmROC)
```

```
## Area under the curve: 0.743
```
