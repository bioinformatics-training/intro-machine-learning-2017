# Support vector machines {#svm}

## Introduction



<!-- Matt -->

<!--REPEAT CLASSIFICATION EXAMPLE FOR SERUM PROTEOMICS -->

<!-- regression and classification -->





## Classification


### Example - simulated data


```r
linearly_separable <- read.csv("data/sim_data_svm/linearly_separable.csv", header=F)
circles <- read.csv("data/sim_data_svm/circles.csv", header=F)
moons <- read.csv("data/sim_data_svm/moons.csv", header=F)
```

### Example -  cell segmentation data
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


Setup parallel processing

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
getDoParWorkers()
```

```
## [1] 2
```

MUST GENERATE A LIST OF SEEDS IF WE USE PARALLEL PROCESSING, FOR REPRODUCIBILITY

We will return to the cell segmentation data set [@Hill2007] we looked at earlier.

```r
data(segmentationData)
segClass <- segmentationData$Class
segData <- segmentationData[,4:61]
set.seed(42)
trainIndex <- createDataPartition(y=segClass, times=1, p=0.5, list=F)
segDataTrain <- segData[trainIndex,]
segDataTest <- segData[-trainIndex,]
segClassTrain <- segClass[trainIndex]
segClassTest <- segClass[-trainIndex]

transformations <- preProcess(segDataTrain, 
                              method=c("YeoJohnson", "center", "scale", "corr"),
                              cutoff=0.75)
segDataTrain <- predict(transformations, segDataTrain)
```


We will pass the twoClassSummary function into model training through **trainControl**. Additionally we would like the model to predict class probabilities so that we can calculate the ROC curve, so we use the **classProbs** option. 

```r
cvCtrl <- trainControl(method = "repeatedcv", repeats = 3,
                        summaryFunction = twoClassSummary,
                        classProbs = TRUE)
```

Tune SVM over the cost parameter

```r
svmTune <- train(x = segDataTrain,
                 y = segClassTrain,
                 method = "svmRadial",
                 # The default grid of cost parameters go from 2^-2,
                 # 0.5 to 1,
                 # Well fit 9 values in that sequence via the tuneLength
                 # argument.
                 tuneLength = 9,
                 ## Also add options from preProcess here too
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
## Resampling: Cross-Validated (10 fold, repeated 3 times) 
## Summary of sample sizes: 909, 909, 909, 909, 909, 909, ... 
## Resampling results across tuning parameters:
## 
##   C      ROC        Sens       Spec     
##    0.25  0.8831481  0.8630769  0.7083333
##    0.50  0.8885328  0.8666667  0.7009259
##    1.00  0.8934473  0.8666667  0.7259259
##    2.00  0.8941880  0.8707692  0.7361111
##    4.00  0.8879915  0.8661538  0.7120370
##    8.00  0.8767094  0.8610256  0.6898148
##   16.00  0.8617379  0.8589744  0.6509259
##   32.00  0.8515527  0.8528205  0.6425926
##   64.00  0.8494302  0.8553846  0.6370370
## 
## Tuning parameter 'sigma' was held constant at a value of 0.02388876
## ROC was used to select the optimal model using  the largest value.
## The final values used for the model were sigma = 0.02388876 and C = 2.
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
##  Hyperparameter : sigma =  0.0238887592749294 
## 
## Number of Support Vectors : 496 
## 
## Objective Function Value : -675.9703 
## Training error : 0.115842 
## Probability model included.
```


SVM accuracy profile


```r
plot(svmTune, metric = "ROC", scales = list(x = list(log =
2)))
```

<div class="figure" style="text-align: center">
<img src="06-svm_files/figure-html/svmAccuracyProfile-1.png" alt="SVM accuracy profile." width="80%" />
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
##         PS 567 113
##         WS  83 246
##                                         
##                Accuracy : 0.8057        
##                  95% CI : (0.78, 0.8297)
##     No Information Rate : 0.6442        
##     P-Value [Acc > NIR] : < 2e-16       
##                                         
##                   Kappa : 0.5682        
##  Mcnemar's Test P-Value : 0.03832       
##                                         
##             Sensitivity : 0.8723        
##             Specificity : 0.6852        
##          Pos Pred Value : 0.8338        
##          Neg Pred Value : 0.7477        
##              Prevalence : 0.6442        
##          Detection Rate : 0.5619        
##    Detection Prevalence : 0.6739        
##       Balanced Accuracy : 0.7788        
##                                         
##        'Positive' Class : PS            
## 
```

<!--GENERATE TWO  DIFFERENT MODELS AND COMPARE -->
LINEAR
RBM


<!--REPEAT CLASSIFICATION EXAMPLE FOR SERUM PROTEOMICS -->

## Exercises

Solutions to exercises can be found in appendix \@ref(solutions-svm)

<!--

## Serum proteomics

```r
centre1 <- read.csv("data/serum_proteomics/male_centre1.csv")
centre2 <- read.csv("data/serum_proteomics/male_centre2.csv")

diag_cent1 <- centre1$Diagnostic_group
prot_cent1 <- centre1[,2:18]

diag_cent2 <- centre2$Diagnostic_group
prot_cent2 <- centre2[,2:18]

# featurePlot(x=prot_cent1, y=diag_cent1, plot="pairs")

transformations <- preProcess(prot_cent1, 
                              method=c("center", "scale"),
                              cutoff=0.75)
prot_cent1 <- predict(transformations, prot_cent1)

svmTune <- train(x = prot_cent1,
                 y = diag_cent1,
                 method = "svmRadial",
                 # The default grid of cost parameters go from 2^-2,
                 # 0.5 to 1,
                 # Well fit 9 values in that sequence via the tuneLength
                 # argument.
                 tuneLength = 9,
                 ## Also add options from preProcess here too
                 preProc = c("center", "scale"),
                 metric = "ROC",
                 trControl = cvCtrl)

svmTune
```

```
## Support Vector Machines with Radial Basis Function Kernel 
## 
## 73 samples
## 17 predictors
##  2 classes: 'NC', 'SZ' 
## 
## Pre-processing: centered (17), scaled (17) 
## Resampling: Cross-Validated (10 fold, repeated 3 times) 
## Summary of sample sizes: 66, 66, 66, 64, 65, 66, ... 
## Resampling results across tuning parameters:
## 
##   C      ROC        Sens       Spec     
##    0.25  0.6447222  0.4805556  0.6850000
##    0.50  0.6725000  0.5083333  0.7766667
##    1.00  0.6548611  0.4222222  0.7566667
##    2.00  0.6725000  0.4416667  0.8200000
##    4.00  0.6491667  0.3361111  0.8366667
##    8.00  0.6993056  0.3805556  0.7950000
##   16.00  0.7075000  0.4138889  0.7966667
##   32.00  0.7111111  0.4611111  0.7800000
##   64.00  0.7138889  0.4833333  0.7550000
## 
## Tuning parameter 'sigma' was held constant at a value of 0.04808532
## ROC was used to select the optimal model using  the largest value.
## The final values used for the model were sigma = 0.04808532 and C = 64.
```

```r
svmTune$finalModel
```

```
## Support Vector Machine object of class "ksvm" 
## 
## SV type: C-svc  (classification) 
##  parameter : cost C = 64 
## 
## Gaussian Radial Basis kernel function. 
##  Hyperparameter : sigma =  0.048085319074945 
## 
## Number of Support Vectors : 55 
## 
## Objective Function Value : -137.6085 
## Training error : 0 
## Probability model included.
```

-->
