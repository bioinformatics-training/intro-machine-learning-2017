# Solutions ch. 7 - Support vector machines {#solutions-svm}

Solutions to exercises of chapter \@ref(svm).

## Exercise 1

## Exercise 2

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

Tune SVM over the cost parameter. The default grid of cost parameters start at 0.25 and double at each iteration. Choosing ```tuneLength = 9``` will give us cost parameters of 0.25, 0.5, 1, 2, 4, 8, 16, 32 and 64. The train function will calculate the value of $$sigma$$ from the data.

```r
svmTune <- train(x = segDataTrain,
                 y = segClassTrain,
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
## 1010 samples
##   27 predictor
##    2 classes: 'PS', 'WS' 
## 
## Pre-processing: centered (27), scaled (27) 
## Resampling: Cross-Validated (5 fold, repeated 5 times) 
## Summary of sample sizes: 808, 808, 808, 808, 808, 808, ... 
## Resampling results across tuning parameters:
## 
##   C      ROC        Sens       Spec     
##    0.25  0.8818932  0.8606154  0.7055556
##    0.50  0.8871453  0.8673846  0.7044444
##    1.00  0.8921838  0.8698462  0.7155556
##    2.00  0.8925641  0.8689231  0.7255556
##    4.00  0.8857308  0.8658462  0.6950000
##    8.00  0.8747778  0.8615385  0.6755556
##   16.00  0.8613462  0.8612308  0.6500000
##   32.00  0.8518504  0.8572308  0.6338889
##   64.00  0.8511709  0.8535385  0.6405556
## 
## Tuning parameter 'sigma' was held constant at a value of 0.02492821
## ROC was used to select the optimal model using  the largest value.
## The final values used for the model were sigma = 0.02492821 and C = 2.
```


```r
svmTune$finalModel
```

```
## Support Vector Machine object of class "ksvm" 
## 
## SV type: C-svc  (classification) 
##  parameter : cost C = 2 
## 
## Gaussian Radial Basis kernel function. 
##  Hyperparameter : sigma =  0.0249282111402021 
## 
## Number of Support Vectors : 498 
## 
## Objective Function Value : -667.6455 
## Training error : 0.112871 
## Probability model included.
```

SVM accuracy profile

```r
plot(svmTune, metric = "ROC", scales = list(x = list(log =2)))
```

<div class="figure" style="text-align: center">
<img src="17-solutions-svm_files/figure-html/svmAccuracyProfile-1.png" alt="SVM accuracy profile." width="80%" />
<p class="caption">(\#fig:svmAccuracyProfile)SVM accuracy profile.</p>
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
##         PS 565 112
##         WS  85 247
##                                           
##                Accuracy : 0.8048          
##                  95% CI : (0.7789, 0.8288)
##     No Information Rate : 0.6442          
##     P-Value [Acc > NIR] : < 2e-16         
##                                           
##                   Kappa : 0.5668          
##  Mcnemar's Test P-Value : 0.06397         
##                                           
##             Sensitivity : 0.8692          
##             Specificity : 0.6880          
##          Pos Pred Value : 0.8346          
##          Neg Pred Value : 0.7440          
##              Prevalence : 0.6442          
##          Detection Rate : 0.5600          
##    Detection Prevalence : 0.6710          
##       Balanced Accuracy : 0.7786          
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
##          PS         WS
## 1 0.3616549 0.63834514
## 2 0.9006741 0.09932587
## 3 0.7537149 0.24628514
## 4 0.7098942 0.29010578
## 5 0.9364351 0.06356489
## 6 0.7108889 0.28911114
```

Build a ROC curve

```r
svmROC <- roc(segClassTest, svmProbs[,"PS"])
auc(svmROC)
```

```
## Area under the curve: 0.8841
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
<img src="17-solutions-svm_files/figure-html/svmROCcurve-1.png" alt="SVM accuracy profile." width="80%" />
<p class="caption">(\#fig:svmROCcurve)SVM accuracy profile.</p>
</div>

Calculate area under ROC curve

```r
auc(svmROC)
```

```
## Area under the curve: 0.8841
```
