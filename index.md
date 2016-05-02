# Analysis and Prediction of Weightlifting Competency (PML)
John C McDavid  
May 1, 2016  

# Objective  
The objective of this study is to develop a practical machine learning algorithm that will predict the manner (the correct way and five incorrect ways) in which volunteers lift dumbbells in a weightlifting exercise. After using a dataset to train the algorithm, the accuracy of the algorithm will be measured against a test dataset.  

## Overview  
A research study was done in furtherance of human activity recognition research to develop an algorithm that could discriminate between the various ways a person might perform a weightlifting activity. Six young healthy participants were asked to execute a series of sets of repetitions of the Unilateral Dumbbell Biceps Curl in five incorrect ways and one correct way. Various measurements were taken during each set of repetitions along with the manner in which the volunteers were asked to perform the exercise.  
The five variations of doing the weightlifting exercise were:  
  A correct way  
  B throw elbows to front  
  C lift halfway  
  D lower halfway  
  E throw hips to front  

## Citation and reference  
This anlysis is this paper uses data from the following research:
Ugulino, W.; Cardador, D.; Vega, K.; Velloso, E.; Milidiu, R.; Fuks, H. Wearable Computing: Accelerometers' Data Classification of Body Postures and Movements. Proceedings of 21st Brazilian Symposium on Artificial Intelligence. Advances in Artificial Intelligence - SBIA 2012. In: Lecture Notes in Computer Science. , pp. 52-61. Curitiba, PR: Springer Berlin / Heidelberg, 2012. ISBN 978-3-642-34458-9. DOI: 10.1007/978-3-642-34459-6_6. Read more at http://groupware.les.inf.puc-rio.br/har#ixzz47MfWwhbW.  

## Approach  
In this paper we will develop an algorithm to discriminate between the variosu ways the researchers had the particpants periform the weightlifting exercise. We ultimately settle on using a random forest method as the best method to develop an algorithm to meet the objective but we also explore the possibilities of using random forest with the data centered and scaled, gbm, and lda. The alternative models were all inferior or did not add appreciable accuracy.  
Cross-validation is handled internal to the caret::train function through train control.  
We also use parallel processing to help with run time.  

## Data  
The training and test data for the project are from:  
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv  
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv  

We initialize and clear the workspace, load and read in the data, and load the necessary R libaries (we assume the libararies are installed).  


```r
setwd("C:\\Users\\jcmcd\\Coursera - Practical Machine Learning")
rm(list=ls())

# libraries
library(caret); library(ggplot2); library(lattice); library(rpart); library(rpart.plot)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```
## Warning: package 'rpart.plot' was built under R version 3.2.5
```

```r
library(doParallel); library(parallel); library(randomForest)
```

```
## Warning: package 'doParallel' was built under R version 3.2.5
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

```
## Warning: package 'randomForest' was built under R version 3.2.5
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
library(rattle); library(corrplot)
```

```
## Warning: package 'rattle' was built under R version 3.2.5
```

```
## Rattle: A free graphical interface for data mining with R.
## Version 4.1.0 Copyright (c) 2006-2015 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```
## Warning: package 'corrplot' was built under R version 3.2.5
```

```r
# load data
# training
fileUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv?accessType=DOWNLOAD"
download.file(fileUrl, destfile="./data/pml-train.csv")
Train <- read.csv("./data/pml-train.csv")
# testing
fileUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv?accessType=DOWNLOAD"
download.file(fileUrl, destfile="./data/pml-test.csv")
Test <- read.csv("./data/pml-test.csv")
```

## Exploratory Analysis  
We explore the dataset.  

```r
unique(Train$classe)
```

```
## [1] A B C D E
## Levels: A B C D E
```

```r
dim(Train); dim(Test)
```

```
## [1] 19622   160
```

```
## [1]  20 160
```

```r
sum(Train$classe == "A")
```

```
## [1] 5580
```

```r
sum(Train$classe == "B")
```

```
## [1] 3797
```

```r
sum(Train$classe == "C")
```

```
## [1] 3422
```

```r
sum(Train$classe == "D")
```

```
## [1] 3216
```

```r
sum(Train$classe == "E")
```

```
## [1] 3607
```

```r
sum(5580, 3797, 3422, 3216, 3607)
```

```
## [1] 19622
```

```r
names(Train)
```

```
##   [1] "X"                        "user_name"               
##   [3] "raw_timestamp_part_1"     "raw_timestamp_part_2"    
##   [5] "cvtd_timestamp"           "new_window"              
##   [7] "num_window"               "roll_belt"               
##   [9] "pitch_belt"               "yaw_belt"                
##  [11] "total_accel_belt"         "kurtosis_roll_belt"      
##  [13] "kurtosis_picth_belt"      "kurtosis_yaw_belt"       
##  [15] "skewness_roll_belt"       "skewness_roll_belt.1"    
##  [17] "skewness_yaw_belt"        "max_roll_belt"           
##  [19] "max_picth_belt"           "max_yaw_belt"            
##  [21] "min_roll_belt"            "min_pitch_belt"          
##  [23] "min_yaw_belt"             "amplitude_roll_belt"     
##  [25] "amplitude_pitch_belt"     "amplitude_yaw_belt"      
##  [27] "var_total_accel_belt"     "avg_roll_belt"           
##  [29] "stddev_roll_belt"         "var_roll_belt"           
##  [31] "avg_pitch_belt"           "stddev_pitch_belt"       
##  [33] "var_pitch_belt"           "avg_yaw_belt"            
##  [35] "stddev_yaw_belt"          "var_yaw_belt"            
##  [37] "gyros_belt_x"             "gyros_belt_y"            
##  [39] "gyros_belt_z"             "accel_belt_x"            
##  [41] "accel_belt_y"             "accel_belt_z"            
##  [43] "magnet_belt_x"            "magnet_belt_y"           
##  [45] "magnet_belt_z"            "roll_arm"                
##  [47] "pitch_arm"                "yaw_arm"                 
##  [49] "total_accel_arm"          "var_accel_arm"           
##  [51] "avg_roll_arm"             "stddev_roll_arm"         
##  [53] "var_roll_arm"             "avg_pitch_arm"           
##  [55] "stddev_pitch_arm"         "var_pitch_arm"           
##  [57] "avg_yaw_arm"              "stddev_yaw_arm"          
##  [59] "var_yaw_arm"              "gyros_arm_x"             
##  [61] "gyros_arm_y"              "gyros_arm_z"             
##  [63] "accel_arm_x"              "accel_arm_y"             
##  [65] "accel_arm_z"              "magnet_arm_x"            
##  [67] "magnet_arm_y"             "magnet_arm_z"            
##  [69] "kurtosis_roll_arm"        "kurtosis_picth_arm"      
##  [71] "kurtosis_yaw_arm"         "skewness_roll_arm"       
##  [73] "skewness_pitch_arm"       "skewness_yaw_arm"        
##  [75] "max_roll_arm"             "max_picth_arm"           
##  [77] "max_yaw_arm"              "min_roll_arm"            
##  [79] "min_pitch_arm"            "min_yaw_arm"             
##  [81] "amplitude_roll_arm"       "amplitude_pitch_arm"     
##  [83] "amplitude_yaw_arm"        "roll_dumbbell"           
##  [85] "pitch_dumbbell"           "yaw_dumbbell"            
##  [87] "kurtosis_roll_dumbbell"   "kurtosis_picth_dumbbell" 
##  [89] "kurtosis_yaw_dumbbell"    "skewness_roll_dumbbell"  
##  [91] "skewness_pitch_dumbbell"  "skewness_yaw_dumbbell"   
##  [93] "max_roll_dumbbell"        "max_picth_dumbbell"      
##  [95] "max_yaw_dumbbell"         "min_roll_dumbbell"       
##  [97] "min_pitch_dumbbell"       "min_yaw_dumbbell"        
##  [99] "amplitude_roll_dumbbell"  "amplitude_pitch_dumbbell"
## [101] "amplitude_yaw_dumbbell"   "total_accel_dumbbell"    
## [103] "var_accel_dumbbell"       "avg_roll_dumbbell"       
## [105] "stddev_roll_dumbbell"     "var_roll_dumbbell"       
## [107] "avg_pitch_dumbbell"       "stddev_pitch_dumbbell"   
## [109] "var_pitch_dumbbell"       "avg_yaw_dumbbell"        
## [111] "stddev_yaw_dumbbell"      "var_yaw_dumbbell"        
## [113] "gyros_dumbbell_x"         "gyros_dumbbell_y"        
## [115] "gyros_dumbbell_z"         "accel_dumbbell_x"        
## [117] "accel_dumbbell_y"         "accel_dumbbell_z"        
## [119] "magnet_dumbbell_x"        "magnet_dumbbell_y"       
## [121] "magnet_dumbbell_z"        "roll_forearm"            
## [123] "pitch_forearm"            "yaw_forearm"             
## [125] "kurtosis_roll_forearm"    "kurtosis_picth_forearm"  
## [127] "kurtosis_yaw_forearm"     "skewness_roll_forearm"   
## [129] "skewness_pitch_forearm"   "skewness_yaw_forearm"    
## [131] "max_roll_forearm"         "max_picth_forearm"       
## [133] "max_yaw_forearm"          "min_roll_forearm"        
## [135] "min_pitch_forearm"        "min_yaw_forearm"         
## [137] "amplitude_roll_forearm"   "amplitude_pitch_forearm" 
## [139] "amplitude_yaw_forearm"    "total_accel_forearm"     
## [141] "var_accel_forearm"        "avg_roll_forearm"        
## [143] "stddev_roll_forearm"      "var_roll_forearm"        
## [145] "avg_pitch_forearm"        "stddev_pitch_forearm"    
## [147] "var_pitch_forearm"        "avg_yaw_forearm"         
## [149] "stddev_yaw_forearm"       "var_yaw_forearm"         
## [151] "gyros_forearm_x"          "gyros_forearm_y"         
## [153] "gyros_forearm_z"          "accel_forearm_x"         
## [155] "accel_forearm_y"          "accel_forearm_z"         
## [157] "magnet_forearm_x"         "magnet_forearm_y"        
## [159] "magnet_forearm_z"         "classe"
```

## Cleaning  
We have to remove columns that have missing data and remove seven other cols (X, user name, 3 timestamps, 2 "windows") that aren't useful. This results in 53 remaining variables, including the outcome variable "classe."  
We then apply the same removal to the test dataset and check to make sure there are no additional cols in the test 
dataset that need to be removed (there aren't any).  
Finally, we make sure the fields in the train and test datasets are the same (except for the outcome variable in the train set and the problem_id variable in the test set.)  


```r
# have to elim cols that have NA or ""
msgcols2 <- sapply(Train, function(x) any(is.na(x) | x == ""))
sum(msgcols2)
```

```
## [1] 100
```

```r
TrainCln1 <- names(msgcols2)[!msgcols2]
TrainCln2 <- TrainCln1[-c(1:7)]

Train2 <- Train[,TrainCln2]
dim(Train2)
```

```
## [1] 19622    53
```

```r
# do same clean on test, make sure nothing additional is msg in test
Test2 <- Test[,c(TrainCln2[-53],"problem_id")]
dim(Test2)
```

```
## [1] 20 53
```

```r
# [1] 20 53
msgcols2test <- sapply(Test2, function(x) any(is.na(x) | x == ""))  # ck for additional msg cols in Test
sum(msgcols2test)  # any addit'l cols w missing vals?
```

```
## [1] 0
```

```r
sum(names(Train2) == names(Test2))
```

```
## [1] 52
```

## Partitioning  
We set a seed and partition the cleaned train dataset into training and testing datasets using a 70/30 split.  The original test dataset is held out from the aanlysis.  


```r
set.seed(526526)
inTrain <- createDataPartition(y = Train2$classe,
                               p = 0.7, list = FALSE)
TrainClnTrain <- Train2[inTrain,]
TrainClnTest <- Train2[-inTrain,]
```

## Test for near zero variance of predictors and linear combinations of cols.

```r
# test nzv nearZeroVar on training
nzv <- nearZeroVar(TrainClnTrain, saveMetrics = TRUE)
# nzv
sum(nzv[,3],nzv[,4])    # any zero or near zero variances?
```

```
## [1] 0
```

```r
# look for linear combins
findLinearCombos(Train2[,-53])
```

```
## $linearCombos
## list()
## 
## $remove
## NULL
```

We now have cleaned Train2, Test2 datasets + Train2 partitioned to TrainClnTrain, TrainClnTest

```r
dim(Train2)
```

```
## [1] 19622    53
```

```r
dim(TrainClnTrain)
```

```
## [1] 13737    53
```

```r
dim(TrainClnTest)
```

```
## [1] 5885   53
```

## More Exploration
As an initial exploration, we fit a random forest tree and plot a dendrogram of what we might expect. We also do a correlation matrix plot.  

```r
cluster <- makeCluster(detectCores() - 1)    # convention is to leave 1 core for OS
registerDoParallel(cluster)

mstart <- Sys.time()
Agraph <- rpart(classe ~ ., data = TrainClnTrain, method = "class")
mend <- Sys.time()
c(mstart, mend)
```

```
## [1] "2016-05-01 22:50:37 EDT" "2016-05-01 22:50:40 EDT"
```

```r
stopCluster(cluster)   # de-register parallel process cluster

prp(Agraph)
```

![](PML-Project_v2_files/figure-html/unnamed-chunk-7-1.png)<!-- -->

```r
# fancyRpartPlot(Agraph)      # works but tooo many bottow rows....prp more readable

cPlot <- cor(TrainClnTrain[,-53])
corrplot(cPlot, method = "color")
```

![](PML-Project_v2_files/figure-html/unnamed-chunk-7-2.png)<!-- -->

# Main Model 4.
After model testing and review we settle on a random forest model to predict the manner in which the weightlifting exercise was performed. The accuracy was best using this model.  Centering and scaling was tested but found not to give more than a negligible improvement so wasn't done in the final model used.  
Alternative models were tested and are briefly described below.
Cross validation was handled internal to training through train control.  The out of sample error rate (OOB) was 0.72%.  Our out of sample error rate as measured on the test partition of the original train dataset is 0.66%.  
We also look at variable importance.  
Finally, we make our predictions on the original testing dataset provided in the downloads.  

```r
library(doParallel)
library(parallel)
cluster <- makeCluster(detectCores() - 1)    # convention is to leave 1 core for OS
registerDoParallel(cluster)

set.seed(526)
ctrl <- trainControl(method="cv", number = 10, allowParallel = TRUE)

mstart <- Sys.time()
modFit4 <- train(classe ~ ., method = "rf", data = TrainClnTrain, trControl = ctrl)
mend <- Sys.time()
c(mstart, mend)
```

```
## [1] "2016-05-01 22:50:42 EDT" "2016-05-01 23:07:55 EDT"
```

```r
stopCluster(cluster)

modFit4
```

```
## Random Forest 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 12363, 12361, 12365, 12363, 12362, 12363, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9919915  0.9898685  0.001914153  0.002422719
##   27    0.9912645  0.9889491  0.002590485  0.003277818
##   52    0.9884259  0.9853582  0.002898504  0.003667845
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

```r
modFit4$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 0.72%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3903    3    0    0    0 0.0007680492
## B   16 2634    8    0    0 0.0090293454
## C    0   20 2374    2    0 0.0091819699
## D    0    0   40 2211    1 0.0182060391
## E    0    1    2    6 2516 0.0035643564
```

```r
varImp(modFit4)
```

```
## rf variable importance
## 
##   only 20 most important variables shown (out of 52)
## 
##                   Overall
## roll_belt          100.00
## yaw_belt            79.34
## magnet_dumbbell_z   70.02
## pitch_belt          63.02
## magnet_dumbbell_y   60.95
## pitch_forearm       59.14
## roll_forearm        52.85
## magnet_dumbbell_x   52.34
## accel_belt_z        45.76
## roll_dumbbell       43.49
## accel_dumbbell_y    43.31
## magnet_belt_z       42.70
## magnet_belt_y       40.20
## accel_dumbbell_z    38.25
## roll_arm            35.82
## accel_forearm_x     32.20
## gyros_belt_z        32.09
## yaw_dumbbell        29.61
## accel_dumbbell_x    29.28
## gyros_dumbbell_y    28.72
```

```r
# order(varImp(modFit4), decreasing = T)

varImp(modFit4, scale = "TRUE")
```

```
## rf variable importance
## 
##   only 20 most important variables shown (out of 52)
## 
##                   Overall
## roll_belt          100.00
## yaw_belt            79.34
## magnet_dumbbell_z   70.02
## pitch_belt          63.02
## magnet_dumbbell_y   60.95
## pitch_forearm       59.14
## roll_forearm        52.85
## magnet_dumbbell_x   52.34
## accel_belt_z        45.76
## roll_dumbbell       43.49
## accel_dumbbell_y    43.31
## magnet_belt_z       42.70
## magnet_belt_y       40.20
## accel_dumbbell_z    38.25
## roll_arm            35.82
## accel_forearm_x     32.20
## gyros_belt_z        32.09
## yaw_dumbbell        29.61
## accel_dumbbell_x    29.28
## gyros_dumbbell_y    28.72
```

```r
# order(varImp(modFit4), decreasing = T)

# plot tree

# predict on trainclntest
pred.modFit4 <- predict(modFit4, TrainClnTest)
confusionMatrix(TrainClnTest$classe, pred.modFit4)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    1    0    0    0
##          B    4 1133    2    0    0
##          C    0    8 1016    2    0
##          D    0    0   20  944    0
##          E    0    0    0    2 1080
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9934         
##                  95% CI : (0.991, 0.9953)
##     No Information Rate : 0.285          
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9916         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9976   0.9921   0.9788   0.9958   1.0000
## Specificity            0.9998   0.9987   0.9979   0.9959   0.9996
## Pos Pred Value         0.9994   0.9947   0.9903   0.9793   0.9982
## Neg Pred Value         0.9991   0.9981   0.9955   0.9992   1.0000
## Prevalence             0.2850   0.1941   0.1764   0.1611   0.1835
## Detection Rate         0.2843   0.1925   0.1726   0.1604   0.1835
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9987   0.9954   0.9884   0.9959   0.9998
```

```r
Accuracy <- postResample(pred.modFit4, TrainClnTest$classe)
Accuracy
```

```
##  Accuracy     Kappa 
## 0.9933730 0.9916167
```

```r
# out of sample error is 1 - Accuracy on testing set
1- Accuracy
```

```
##    Accuracy       Kappa 
## 0.006627018 0.008383262
```

```r
# report noth oob and traincln test err rate???

# now predict on Test set fron original download ....P
pred.modFit4Test <- predict(modFit4, Test2)
pred.modFit4Test
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

```r
# end 4 ...works
```

Graphing:  

```r
# 4.1.2 for graphing

cluster <- makeCluster(detectCores() - 1)    # convention is to leave 1 core for OS
registerDoParallel(cluster)

mstart <- Sys.time()
Agraph <- rpart(classe ~ ., data = TrainClnTrain, method = "class")
mend <- Sys.time()
c(mstart, mend)
```

```
## [1] "2016-05-01 23:07:57 EDT" "2016-05-01 23:07:59 EDT"
```

```r
stopCluster(cluster)   # de-register parallel process cluster


prp(Agraph)     # works
```

![](PML-Project_v2_files/figure-html/unnamed-chunk-9-1.png)<!-- -->

```r
fancyRpartPlot(Agraph)      # works but tooo many bottow rows....prp more readable
```

```
## Warning: labs do not fit even at cex 0.15, there may be some overplotting
```

![](PML-Project_v2_files/figure-html/unnamed-chunk-9-2.png)<!-- -->

## Results/Conclusions
The random forest model with internal cross validation proved to be teh most accurate model of all the model types tested. The predictions made on the test dataset are reported above.  

## End of results and model

## Model Alternatives
Several models were tested as alternative possibilities, none performing as well as the above. Here we show the gbm, lda, and rf with centering and scaling models.

### GBM model
This model is not as good as the chosen random forest model; the out of sample error rate is 4%.  

```r
# -------------------------
# 4.2 w gbm (c/p from 4 w modif)
library(doParallel)
library(parallel)
cluster <- makeCluster(detectCores() - 1)    # convention is to leave 1 core for OS
registerDoParallel(cluster)

set.seed(526)
ctrl <- trainControl(method="cv", number = 10, allowParallel = TRUE)

mstart <- Sys.time()
modFit4.2 <- train(classe ~ ., method = "gbm", data = TrainClnTrain, trControl = ctrl)    # method gbm
```

```
## Loading required package: gbm
```

```
## Warning: package 'gbm' was built under R version 3.2.5
```

```
## Loading required package: survival
```

```
## 
## Attaching package: 'survival'
```

```
## The following object is masked from 'package:caret':
## 
##     cluster
```

```
## Loading required package: splines
```

```
## Loaded gbm 2.1.1
```

```
## Loading required package: plyr
```

```
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2320
##      2        1.4609             nan     0.1000    0.1617
##      3        1.3602             nan     0.1000    0.1232
##      4        1.2822             nan     0.1000    0.1097
##      5        1.2137             nan     0.1000    0.0856
##      6        1.1571             nan     0.1000    0.0827
##      7        1.1060             nan     0.1000    0.0666
##      8        1.0637             nan     0.1000    0.0631
##      9        1.0229             nan     0.1000    0.0478
##     10        0.9915             nan     0.1000    0.0566
##     20        0.7544             nan     0.1000    0.0234
##     40        0.5258             nan     0.1000    0.0115
##     60        0.4071             nan     0.1000    0.0078
##     80        0.3233             nan     0.1000    0.0028
##    100        0.2668             nan     0.1000    0.0028
##    120        0.2229             nan     0.1000    0.0023
##    140        0.1903             nan     0.1000    0.0014
##    150        0.1767             nan     0.1000    0.0014
```

```r
mend <- Sys.time()
# 
stopCluster(cluster)
c(mstart, mend)
```

```
## [1] "2016-05-01 23:08:02 EDT" "2016-05-01 23:13:31 EDT"
```

```r
modFit4.2
```

```
## Stochastic Gradient Boosting 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 12363, 12361, 12365, 12363, 12362, 12363, ... 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy   Kappa      Accuracy SD
##   1                   50      0.7553998  0.6899370  0.013111832
##   1                  100      0.8210647  0.7734430  0.011702037
##   1                  150      0.8541860  0.8153523  0.009987297
##   2                   50      0.8546992  0.8158076  0.009902794
##   2                  100      0.9071859  0.8825414  0.008108452
##   2                  150      0.9322272  0.9142251  0.006800047
##   3                   50      0.8946610  0.8666288  0.008316264
##   3                  100      0.9429254  0.9277652  0.005435254
##   3                  150      0.9616352  0.9514571  0.004014674
##   Kappa SD   
##   0.016773138
##   0.014878579
##   0.012719421
##   0.012632308
##   0.010294598
##   0.008648046
##   0.010613669
##   0.006907220
##   0.005096266
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## Accuracy was used to select the optimal model using  the largest value.
## The final values used for the model were n.trees = 150,
##  interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.
```

```r
modFit4.2$finalModel
```

```
## A gradient boosted model with multinomial loss function.
## 150 iterations were performed.
## There were 52 predictors of which 44 had non-zero influence.
```

```r
postResample(predict(modFit4.2,TrainClnTrain),TrainClnTrain$classe)
```

```
##  Accuracy     Kappa 
## 0.9736478 0.9666614
```

```r
varImp(modFit4.2)
```

```
## gbm variable importance
## 
##   only 20 most important variables shown (out of 52)
## 
##                   Overall
## roll_belt         100.000
## pitch_forearm      50.563
## yaw_belt           38.251
## magnet_dumbbell_z  33.907
## roll_forearm       23.891
## magnet_belt_z      23.267
## magnet_dumbbell_y  22.965
## roll_dumbbell      16.978
## gyros_belt_z       15.383
## accel_forearm_x    11.997
## pitch_belt         11.128
## accel_forearm_z    10.846
## gyros_dumbbell_y   10.118
## accel_dumbbell_x    8.275
## yaw_arm             7.652
## accel_dumbbell_y    7.644
## magnet_forearm_z    7.322
## magnet_arm_z        7.282
## magnet_belt_x       5.100
## roll_arm            5.079
```

```r
# order(varImp(modFit4.2), decreasing = T)

varImp(modFit4.2, scale = "TRUE")
```

```
## gbm variable importance
## 
##   only 20 most important variables shown (out of 52)
## 
##                   Overall
## roll_belt         100.000
## pitch_forearm      50.563
## yaw_belt           38.251
## magnet_dumbbell_z  33.907
## roll_forearm       23.891
## magnet_belt_z      23.267
## magnet_dumbbell_y  22.965
## roll_dumbbell      16.978
## gyros_belt_z       15.383
## accel_forearm_x    11.997
## pitch_belt         11.128
## accel_forearm_z    10.846
## gyros_dumbbell_y   10.118
## accel_dumbbell_x    8.275
## yaw_arm             7.652
## accel_dumbbell_y    7.644
## magnet_forearm_z    7.322
## magnet_arm_z        7.282
## magnet_belt_x       5.100
## roll_arm            5.079
```

```r
# order(varImp(modFit4.2), decreasing = T)

# predict on trainclntest
pred.modFit4.2 <- predict(modFit4.2, TrainClnTest)
confusionMatrix(TrainClnTest$classe, pred.modFit4.2)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1651   12    7    4    0
##          B   49 1056   34    0    0
##          C    0   29  986    9    2
##          D    0    2   45  912    5
##          E    2   10    9   20 1041
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9594         
##                  95% CI : (0.954, 0.9643)
##     No Information Rate : 0.2892         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9486         
##  Mcnemar's Test P-Value : 4.567e-14      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9700   0.9522   0.9121   0.9651   0.9933
## Specificity            0.9945   0.9826   0.9917   0.9895   0.9915
## Pos Pred Value         0.9863   0.9271   0.9610   0.9461   0.9621
## Neg Pred Value         0.9879   0.9888   0.9804   0.9933   0.9985
## Prevalence             0.2892   0.1884   0.1837   0.1606   0.1781
## Detection Rate         0.2805   0.1794   0.1675   0.1550   0.1769
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9823   0.9674   0.9519   0.9773   0.9924
```

```r
Accuracy <- postResample(pred.modFit4.2, TrainClnTest$classe)
Accuracy
```

```
##  Accuracy     Kappa 
## 0.9593883 0.9486053
```

```r
# out of sample error is 1 - Accuracy on testing set
1- Accuracy
```

```
##   Accuracy      Kappa 
## 0.04061172 0.05139474
```

```r
# now predict on Test set fron original download ....P
pred.modFit4.2Test <- predict(modFit4.2, Test2)
pred.modFit4.2Test
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

```r
# end 4.2
# -------------------------
```

### lda model:  
The linear discriminate analysis model is awful and all characteristics make it a vastly inferior model (out of sample error rate is 29%).  

```r
# 4.3 lda
library(doParallel)
library(parallel)
cluster <- makeCluster(detectCores() - 1)    # convention is to leave 1 core for OS
registerDoParallel(cluster)

set.seed(526)
ctrl <- trainControl(method="cv", number = 10, allowParallel = TRUE)

mstart <- Sys.time()
modFit4.3 <- train(classe ~ ., method = "lda", data = TrainClnTrain, trControl = ctrl)    # method lda
```

```
## Loading required package: MASS
```

```r
mend <- Sys.time()
# 
stopCluster(cluster)
c(mstart, mend)
```

```
## [1] "2016-05-01 23:13:32 EDT" "2016-05-01 23:13:44 EDT"
```

```r
modFit4.3
```

```
## Linear Discriminant Analysis 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 12363, 12361, 12365, 12363, 12362, 12363, ... 
## Resampling results
## 
##   Accuracy  Kappa      Accuracy SD  Kappa SD  
##   0.700154  0.6205345  0.009059321  0.01122097
## 
## 
```

```r
# modFit4.3$finalModel

postResample(predict(modFit4.3,TrainClnTrain),TrainClnTrain$classe)
```

```
##  Accuracy     Kappa 
## 0.7048118 0.6263789
```

```r
# predict on trainclntest
pred.modFit4.3 <- predict(modFit4.3, TrainClnTest)
confusionMatrix(TrainClnTest$classe, pred.modFit4.3)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1366   35  129  136    8
##          B  168  756  120   40   55
##          C  108   89  683  113   33
##          D   50   47  104  718   45
##          E   39  184   98   99  662
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7111          
##                  95% CI : (0.6994, 0.7227)
##     No Information Rate : 0.2941          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6345          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.7891   0.6805   0.6023   0.6492   0.8244
## Specificity            0.9259   0.9198   0.9278   0.9485   0.9174
## Pos Pred Value         0.8160   0.6637   0.6657   0.7448   0.6118
## Neg Pred Value         0.9133   0.9252   0.9072   0.9212   0.9706
## Prevalence             0.2941   0.1888   0.1927   0.1879   0.1364
## Detection Rate         0.2321   0.1285   0.1161   0.1220   0.1125
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.8575   0.8001   0.7650   0.7989   0.8709
```

```r
Accuracy <- postResample(pred.modFit4.3, TrainClnTest$classe)
Accuracy
```

```
##  Accuracy     Kappa 
## 0.7111300 0.6344933
```

```r
# out of sample error is 1 - Accuracy on testing set
1- Accuracy
```

```
##  Accuracy     Kappa 
## 0.2888700 0.3655067
```

```r
# --> 28.89%

# now predict on Test set fron original download ....P
pred.modFit4.3Test <- predict(modFit4.3, Test2)
pred.modFit4.3Test
```

```
##  [1] B A B C C C D D A A D A B A E A A B B B
## Levels: A B C D E
```

```r
# end 4.3

# -------------------------
```

## rf model 4.1 with preprocess Ctr, Scale  
Centering and scaling gives negligible improvement to teh random forest model so was not used.  

```r
library(doParallel)
library(parallel)
cluster <- makeCluster(detectCores() - 1)    # convention is to leave 1 core for OS
registerDoParallel(cluster)

set.seed(526)
ctrl <- trainControl(method="cv", number = 10, allowParallel = TRUE)
mstart <- Sys.time()
modFit4.1 <- train(classe ~ ., method = "rf", data = TrainClnTrain, preProcess=c("center", "scale"), trControl = ctrl)
mend <- Sys.time()
c(mstart, mend)
```

```
## [1] "2016-05-01 23:13:45 EDT" "2016-05-01 23:32:16 EDT"
```

```r
stopCluster(cluster)   # de-register parallel process cluster

modFit4.1
```

```
## Random Forest 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: centered (52), scaled (52) 
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 12363, 12361, 12365, 12363, 12362, 12363, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9917007  0.9895005  0.002282888  0.002889737
##   27    0.9913371  0.9890414  0.002413338  0.003052619
##   52    0.9882808  0.9851739  0.003187619  0.004033552
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

```r
modFit4.1$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 0.75%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3902    4    0    0    0 0.001024066
## B   17 2633    8    0    0 0.009405568
## C    0   20 2373    3    0 0.009599332
## D    0    0   40 2210    2 0.018650089
## E    0    0    3    6 2516 0.003564356
```

```r
# predict on TrainClnTest
pred.modFit4.1 <- predict(modFit4.1, TrainClnTest)
confusionMatrix(TrainClnTest$classe, pred.modFit4.1)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1671    3    0    0    0
##          B    4 1133    2    0    0
##          C    0    8 1016    2    0
##          D    0    0   20  944    0
##          E    0    0    0    2 1080
## 
## Overall Statistics
##                                          
##                Accuracy : 0.993          
##                  95% CI : (0.9906, 0.995)
##     No Information Rate : 0.2846         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9912         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9976   0.9904   0.9788   0.9958   1.0000
## Specificity            0.9993   0.9987   0.9979   0.9959   0.9996
## Pos Pred Value         0.9982   0.9947   0.9903   0.9793   0.9982
## Neg Pred Value         0.9991   0.9977   0.9955   0.9992   1.0000
## Prevalence             0.2846   0.1944   0.1764   0.1611   0.1835
## Detection Rate         0.2839   0.1925   0.1726   0.1604   0.1835
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9984   0.9946   0.9884   0.9959   0.9998
```

```r
Accuracy <- postResample(pred.modFit4.1, TrainClnTest$classe)
Accuracy
```

```
##  Accuracy     Kappa 
## 0.9930331 0.9911872
```

```r
# out of sample error is 1 - Accuracy on testing set
1- Accuracy
```

```
##    Accuracy       Kappa 
## 0.006966865 0.008812828
```

```r
# error went from .71% up to .75%
pred.modFit4.1Test <- predict(modFit4.1, Test2)
pred.modFit4.1Test                   
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

```r
# end 4.1
# -------------------------
```

## 4.1.1 rf model with centering and scaling done a different way...results are the same  
Comments are the same as above.  

```r
library(doParallel)
library(parallel)
cluster <- makeCluster(detectCores() - 1)    # convention is to leave 1 core for OS
registerDoParallel(cluster)

set.seed(526)
ctrl <- trainControl(method="cv", number = 10, allowParallel = TRUE)
preObj <- preProcess(TrainClnTrain[,-53], method = c("center", "scale"))    # step 1
TrainClnTrain.cs <- predict(preObj, TrainClnTrain)                          # step 2a
mstart <- Sys.time()
modFit4.1.1 <- train(classe ~ ., method = "rf", data = TrainClnTrain.cs, trControl = ctrl)    # step 3
mend <- Sys.time()

stopCluster(cluster)   # de-register parallel process cluster
c(mstart, mend)
```

```
## [1] "2016-05-01 23:32:19 EDT" "2016-05-01 23:49:47 EDT"
```

```r
TrainClnTest.cs <- predict(preObj, TrainClnTest)                            # step 2b1
Test2.cs <- predict(preObj, Test2)                                          # step 2b2

modFit4.1.1
```

```
## Random Forest 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 12363, 12361, 12365, 12363, 12362, 12363, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9915550  0.9893161  0.002231228  0.002823904
##   27    0.9917741  0.9895941  0.002328184  0.002945096
##   52    0.9878442  0.9846216  0.003178904  0.004022671
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

```r
modFit4.1.1$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.82%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3901    4    0    0    1 0.001280082
## B   19 2628   11    0    0 0.011286682
## C    0   14 2374    8    0 0.009181970
## D    0    1   36 2212    3 0.017761989
## E    0    2    4    9 2510 0.005940594
```

```r
# similar, sl lower err rate

# predict on TrainClnTest.cs
pred.modFit4.1.1 <- predict(modFit4.1.1, TrainClnTest.cs)
confusionMatrix(TrainClnTest.cs$classe, pred.modFit4.1.1)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    1    0    0    0
##          B    8 1128    3    0    0
##          C    0    6 1016    4    0
##          D    0    0   11  953    0
##          E    0    0    2    2 1078
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9937          
##                  95% CI : (0.9913, 0.9956)
##     No Information Rate : 0.2856          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.992           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9952   0.9938   0.9845   0.9937   1.0000
## Specificity            0.9998   0.9977   0.9979   0.9978   0.9992
## Pos Pred Value         0.9994   0.9903   0.9903   0.9886   0.9963
## Neg Pred Value         0.9981   0.9985   0.9967   0.9988   1.0000
## Prevalence             0.2856   0.1929   0.1754   0.1630   0.1832
## Detection Rate         0.2843   0.1917   0.1726   0.1619   0.1832
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9975   0.9958   0.9912   0.9958   0.9996
```

```r
Accuracy <- postResample(pred.modFit4.1.1, TrainClnTest.cs$classe)
Accuracy
```

```
##  Accuracy     Kappa 
## 0.9937128 0.9920464
```

```r
#  acurr is same, kappa diff in last dec

# out of sample error is 1 - Accuracy on testing set
1- Accuracy
```

```
##    Accuracy       Kappa 
## 0.006287171 0.007953643
```

```r
# error .75%
pred.modFit4.1.1Test.cs <- predict(modFit4.1.1, Test2.cs)
pred.modFit4.1.1Test.cs                   
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

```r
# end 4.1.1
# -------------------------
```

# End Report

```r
Sys.time()
```

```
## [1] "2016-05-01 23:49:47 EDT"
```
