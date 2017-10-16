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

```r
library(e1071)
```

Define a radial SVM using the e1071 library

```r
svmRadialE1071 <- list(
  label = "Support Vector Machines with Radial Kernel - e1071",
  library = "e1071",
  type = c("Regression", "Classification"),
  parameters = data.frame(parameter="cost",
                          class="numeric",
                          label="Cost"),
  grid = function (x, y, len = NULL, search = "grid") 
    {
      if (search == "grid") {
        out <- expand.grid(cost = 2^((1:len) - 3))
      }
      else {
        out <- data.frame(cost = 2^runif(len, min = -5, max = 10))
      }
      out
    },
  loop=NULL,
  fit=function (x, y, wts, param, lev, last, classProbs, ...) 
    {
      if (any(names(list(...)) == "probability") | is.numeric(y)) {
        out <- e1071::svm(x = as.matrix(x), y = y, kernel = "radial", 
                          cost = param$cost, ...)
      }
      else {
        out <- e1071::svm(x = as.matrix(x), y = y, kernel = "radial", 
                          cost = param$cost, probability = classProbs, ...)
      }
      out
    },
  predict = function (modelFit, newdata, submodels = NULL) 
    {
      predict(modelFit, newdata)
    },
  prob = function (modelFit, newdata, submodels = NULL) 
    {
      out <- predict(modelFit, newdata, probability = TRUE)
      attr(out, "probabilities")
    },
  predictors = function (x, ...) 
    {
      out <- if (!is.null(x$terms)) 
        predictors.terms(x$terms)
      else x$xNames
      if (is.null(out)) 
        out <- names(attr(x, "scaling")$x.scale$`scaled:center`)
      if (is.null(out)) 
        out <- NA
      out
    },
  tags = c("Kernel Methods", "Support Vector Machines", "Regression", "Classifier", "Robust Methods"),
  levels = function(x) x$levels,
  sort = function(x)
  {
    x[order(x$cost), ]
  }
)
```

Setup parallel processing

```r
registerDoMC()
getDoParWorkers()
```

```
## [1] 2
```

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

Set seeds for reproducibility (optional). We will be trying 9 values of the tuning parameter with 5 repeats of 10 fold cross-validation, so we need the following list of seeds.

```r
set.seed(42)
seeds <- vector(mode = "list", length = 51)
for(i in 1:50) seeds[[i]] <- sample.int(1000, 9)
seeds[[51]] <- sample.int(1000,1)
```

We will pass the twoClassSummary function into model training through **trainControl**. Additionally we would like the model to predict class probabilities so that we can calculate the ROC curve, so we use the **classProbs** option. 

```r
cvCtrl <- trainControl(method = "repeatedcv", 
                       repeats = 5,
                       number = 10,
                       summaryFunction = twoClassSummary,
                       classProbs = TRUE,
                       seeds=seeds)
```

Tune SVM over the cost parameter. The default grid of cost parameters start at 0.25 and double at each iteration. Choosing ```tuneLength = 9``` will give us cost parameters of 0.25, 0.5, 1, 2, 4, 8, 16, 32 and 64. The train function will calculate an appropriate value of sigma (the kernel parameter) from the data.

```r
svmTune <- train(x = segDataTrain,
                 y = segClassTrain,
                 method = svmRadialE1071,
                 tuneLength = 9,
                 preProc = c("center", "scale"),
                 metric = "ROC",
                 trControl = cvCtrl)

svmTune
```

```
## Support Vector Machines with Radial Kernel - e1071 
## 
## 1010 samples
##   58 predictor
##    2 classes: 'PS', 'WS' 
## 
## Pre-processing: centered (58), scaled (58) 
## Resampling: Cross-Validated (10 fold, repeated 5 times) 
## Summary of sample sizes: 909, 909, 909, 909, 909, 909, ... 
## Resampling results across tuning parameters:
## 
##   cost   ROC        Sens       Spec     
##    0.25  0.8807051  0.8716923  0.6705556
##    0.50  0.8869786  0.8692308  0.7122222
##    1.00  0.8908803  0.8698462  0.7283333
##    2.00  0.8887009  0.8600000  0.7533333
##    4.00  0.8835000  0.8526154  0.7433333
##    8.00  0.8746453  0.8427692  0.7250000
##   16.00  0.8659402  0.8443077  0.7161111
##   32.00  0.8593291  0.8449231  0.7033333
##   64.00  0.8590043  0.8440000  0.6994444
## 
## ROC was used to select the optimal model using  the largest value.
## The final value used for the model was cost = 1.
```


```r
svmTune$finalModel
```

```
## 
## Call:
## svm.default(x = as.matrix(x), y = y, kernel = "radial", cost = param$cost, 
##     probability = classProbs)
## 
## 
## Parameters:
##    SVM-Type:  C-classification 
##  SVM-Kernel:  radial 
##        cost:  1 
##       gamma:  0.01724138 
## 
## Number of Support Vectors:  545
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
#segDataTest <- predict(transformations, segDataTest)
svmPred <- predict(svmTune, segDataTest)
confusionMatrix(svmPred, segClassTest)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction  PS  WS
##         PS 569 104
##         WS  81 255
##                                           
##                Accuracy : 0.8167          
##                  95% CI : (0.7914, 0.8401)
##     No Information Rate : 0.6442          
##     P-Value [Acc > NIR] : <2e-16          
##                                           
##                   Kappa : 0.5942          
##  Mcnemar's Test P-Value : 0.1058          
##                                           
##             Sensitivity : 0.8754          
##             Specificity : 0.7103          
##          Pos Pred Value : 0.8455          
##          Neg Pred Value : 0.7589          
##              Prevalence : 0.6442          
##          Detection Rate : 0.5639          
##    Detection Prevalence : 0.6670          
##       Balanced Accuracy : 0.7928          
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
##           PS         WS
## 3  0.1942982 0.80570183
## 5  0.9357074 0.06429258
## 9  0.7684649 0.23153513
## 10 0.7915982 0.20840184
## 13 0.9445892 0.05541077
## 14 0.7505999 0.24940014
```

Build a ROC curve

```r
svmROC <- roc(segClassTest, svmProbs[,"PS"])
auc(svmROC)
```

```
## Area under the curve: 0.8872
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
## Area under the curve: 0.8872
```

Session info

```r
sessionInfo()
```

```
## R version 3.4.2 (2017-09-28)
## Platform: x86_64-apple-darwin15.6.0 (64-bit)
## Running under: OS X El Capitan 10.11.6
## 
## Matrix products: default
## BLAS: /Library/Frameworks/R.framework/Versions/3.4/Resources/lib/libRblas.0.dylib
## LAPACK: /Library/Frameworks/R.framework/Versions/3.4/Resources/lib/libRlapack.dylib
## 
## locale:
## [1] en_GB.UTF-8/en_GB.UTF-8/en_GB.UTF-8/C/en_GB.UTF-8/en_GB.UTF-8
## 
## attached base packages:
## [1] methods   parallel  stats     graphics  grDevices utils     datasets 
## [8] base     
## 
## other attached packages:
## [1] e1071_1.6-8     pROC_1.10.0     doMC_1.3.4      iterators_1.0.8
## [5] foreach_1.4.3   caret_6.0-77    ggplot2_2.2.1   lattice_0.20-35
## 
## loaded via a namespace (and not attached):
##  [1] purrr_0.2.3        reshape2_1.4.2     kernlab_0.9-25    
##  [4] splines_3.4.2      colorspace_1.3-2   stats4_3.4.2      
##  [7] htmltools_0.3.6    yaml_2.1.14        survival_2.41-3   
## [10] prodlim_1.6.1      rlang_0.1.2        ModelMetrics_1.1.0
## [13] withr_2.0.0        glue_1.1.1         bindrcpp_0.2      
## [16] plyr_1.8.4         bindr_0.1          dimRed_0.1.0      
## [19] lava_1.5.1         robustbase_0.92-7  stringr_1.2.0     
## [22] timeDate_3012.100  munsell_0.4.3      gtable_0.2.0      
## [25] recipes_0.1.0      codetools_0.2-15   evaluate_0.10.1   
## [28] knitr_1.17         class_7.3-14       highr_0.6         
## [31] DEoptimR_1.0-8     Rcpp_0.12.13       scales_0.5.0      
## [34] backports_1.1.1    ipred_0.9-6        CVST_0.2-1        
## [37] digest_0.6.12      stringi_1.1.5      bookdown_0.5      
## [40] dplyr_0.7.4        RcppRoll_0.2.2     ddalpha_1.3.1     
## [43] grid_3.4.2         rprojroot_1.2      tools_3.4.2       
## [46] magrittr_1.5       lazyeval_0.2.0     tibble_1.3.4      
## [49] DRR_0.0.2          pkgconfig_2.0.1    MASS_7.3-47       
## [52] Matrix_1.2-11      lubridate_1.6.0    gower_0.1.2       
## [55] assertthat_0.2.0   rmarkdown_1.6      R6_2.2.2          
## [58] rpart_4.1-11       sfsmisc_1.1-1      nnet_7.3-12       
## [61] nlme_3.1-131       compiler_3.4.2
```
