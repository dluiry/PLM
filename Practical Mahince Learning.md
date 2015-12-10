# Practical machine learning assignment write up
lee ui ryeong  
2015년 11월 21일  

#Plactical Machine Learning-Project Write up
##Introduction
 The goal of this project is to predict the manners what smart advice user did. The target variables is 'classe' variable of training set. The 'classe' variable has 5 categories A, B, C, D and E. This report shows that how predict target variable predicted. we used 3 machine learning algorithms, RandomForest, Linear discriminant analysis and Naive Bayes. After Buiding 3 prediction model,
we choose the model which has the highest accuracy rate.

##About Data

The dataset which used in this project is consist of with 5 classes (sitting-down, standing-up, standing, walking, and sitting) collected on 8 hours of activities of 4 healthy subjects. also this dataset was established with a baseline performance index.

Read more: http://groupware.les.inf.puc-rio.br/har#ixzz3s6Wpactm

##Data resource

-The trining set is available here :

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

-The test set is available here :

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv




##Data loading

```r
Training <-read.csv("pml-training.csv",na.strings=c("NA","#DIV/0!",""))
```

##Looking for Data structure

```r
str(Training, list.len=15)
```

```
## 'data.frame':	19622 obs. of  160 variables:
##  $ X                       : int  1 2 3 4 5 6 7 8 9 10 ...
##  $ user_name               : Factor w/ 6 levels "adelmo","carlitos",..: 2 2 2 2 2 2 2 2 2 2 ...
##  $ raw_timestamp_part_1    : int  1323084231 1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
##  $ raw_timestamp_part_2    : int  788290 808298 820366 120339 196328 304277 368296 440390 484323 484434 ...
##  $ cvtd_timestamp          : Factor w/ 20 levels "02/12/2011 13:32",..: 9 9 9 9 9 9 9 9 9 9 ...
##  $ new_window              : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 1 1 ...
##  $ num_window              : int  11 11 11 12 12 12 12 12 12 12 ...
##  $ roll_belt               : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
##  $ pitch_belt              : num  8.07 8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 ...
##  $ yaw_belt                : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
##  $ total_accel_belt        : int  3 3 3 3 3 3 3 3 3 3 ...
##  $ kurtosis_roll_belt      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_picth_belt     : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_yaw_belt       : logi  NA NA NA NA NA NA ...
##  $ skewness_roll_belt      : num  NA NA NA NA NA NA NA NA NA NA ...
##   [list output truncated]
```

##Drop variables with NA
In this step, we will clean the data and get rid of observations with missing values as well as meaningless variables and time-related variables.

```r
isNA <- apply(Training, 2, function(x) { sum(is.na(x)) })
validTrain <- subset(Training[, which(isNA == 0)], 
                    select=-c(X, user_name, new_window, num_window,                                           raw_timestamp_part_1,
                              raw_timestamp_part_2, cvtd_timestamp))
dim(validTrain)
```

```
## [1] 19622    53
```

```r
names(validTrain)
```

```
##  [1] "roll_belt"            "pitch_belt"           "yaw_belt"            
##  [4] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
##  [7] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
## [10] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
## [13] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
## [16] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
## [19] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
## [22] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
## [25] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
## [28] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
## [31] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
## [34] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
## [37] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
## [40] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
## [43] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
## [46] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
## [49] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
## [52] "magnet_forearm_z"     "classe"
```


##Partioning the dataset into training set and test set

Before we build a prediction model, we partion the training set into two for cross validation purposes. We subsample 60% of the set for training purposes (actual model building), and subsample the 40% of trining set that are validation set,evaluation and accuracy measurement. 


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
inTrain <- createDataPartition(y=validTrain$classe, p=0.6, list=FALSE)
Training <- validTrain[inTrain, ]
Testing <- validTrain[-inTrain, ]
dim(Training)
```

```
## [1] 11776    53
```

##Buidling Prediction Model
##Random Forest
At first,The prediction Model was generated for training set using RandomForest algorithm. 


```r
set.seed(12345)
Cvctrl<-trainControl(method="cv", number=5, allowParallel=T, verbose=T)
modFit <- train(classe~.,data=Training,method="rf",prox=TRUE,trControl=Cvctrl
                ,verbose=F)
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```
## + Fold1: mtry= 2 
## - Fold1: mtry= 2 
## + Fold1: mtry=27 
## - Fold1: mtry=27 
## + Fold1: mtry=52 
## - Fold1: mtry=52 
## + Fold2: mtry= 2 
## - Fold2: mtry= 2 
## + Fold2: mtry=27 
## - Fold2: mtry=27 
## + Fold2: mtry=52 
## - Fold2: mtry=52 
## + Fold3: mtry= 2 
## - Fold3: mtry= 2 
## + Fold3: mtry=27 
## - Fold3: mtry=27 
## + Fold3: mtry=52 
## - Fold3: mtry=52 
## + Fold4: mtry= 2 
## - Fold4: mtry= 2 
## + Fold4: mtry=27 
## - Fold4: mtry=27 
## + Fold4: mtry=52 
## - Fold4: mtry=52 
## + Fold5: mtry= 2 
## - Fold5: mtry= 2 
## + Fold5: mtry=27 
## - Fold5: mtry=27 
## + Fold5: mtry=52 
## - Fold5: mtry=52 
## Aggregating results
## Selecting tuning parameters
## Fitting mtry = 2 on full training set
```

###How accuracy model are?
confusionMatrix() function shows that how prediction model is accuracy.
By using 53 predictors for five classes using cross-validation at a 5-fold an accuracy of 99.07% with a 95% CI [0.9883-0.9927] was achieved accompanied by a Kappa value of 0.9882.



```r
RFpredict <- predict(modFit, newdata=Testing)
confusionMatrix(RFpredict,Testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2225   17    0    0    0
##          B    2 1498   11    0    0
##          C    5    3 1356   36    0
##          D    0    0    1 1250    8
##          E    0    0    0    0 1434
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9894          
##                  95% CI : (0.9869, 0.9916)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9866          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9969   0.9868   0.9912   0.9720   0.9945
## Specificity            0.9970   0.9979   0.9932   0.9986   1.0000
## Pos Pred Value         0.9924   0.9914   0.9686   0.9929   1.0000
## Neg Pred Value         0.9988   0.9968   0.9981   0.9945   0.9988
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2836   0.1909   0.1728   0.1593   0.1828
## Detection Prevalence   0.2858   0.1926   0.1784   0.1605   0.1828
## Balanced Accuracy      0.9969   0.9924   0.9922   0.9853   0.9972
```

##Linear Discriminant Analysis
At first,The prediction Model was generated for training set using linear discriminant analysis algorithm. 



```r
set.seed(12345)
Cvctrl<-trainControl(method="cv", number=5, allowParallel=T, verbose=T)
LDAmodFit <- train(classe~ .,data=Training,method="lda",prox=T,
                   trControl=Cvctrl,verbose=F)
```

```
## Loading required package: MASS
```

```
## + Fold1: parameter=none 
## - Fold1: parameter=none 
## + Fold2: parameter=none 
## - Fold2: parameter=none 
## + Fold3: parameter=none 
## - Fold3: parameter=none 
## + Fold4: parameter=none 
## - Fold4: parameter=none 
## + Fold5: parameter=none 
## - Fold5: parameter=none 
## Aggregating results
## Fitting final model on full training set
```

###How accuracy model are?
confusionMatrix() function shows that how prediction model is accuracy.
By using 53 predictors for five classes using cross-validation at a 5-fold an accuracy of 69.7% with a 95% CI [0.6874-0.7078] was achieved accompanied by a Kappa value of 0.617.


```r
LDApredict <- predict(LDAmodFit, newdata=Testing)
confusionMatrix(LDApredict,Testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1803  243  135   72   50
##          B   60  944  116   61  235
##          C  183  199  893  138  132
##          D  179   44  179  966  136
##          E    7   88   45   49  889
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7004          
##                  95% CI : (0.6901, 0.7105)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.621           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8078   0.6219   0.6528   0.7512   0.6165
## Specificity            0.9109   0.9254   0.8994   0.9180   0.9705
## Pos Pred Value         0.7829   0.6667   0.5780   0.6423   0.8247
## Neg Pred Value         0.9226   0.9107   0.9246   0.9495   0.9183
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2298   0.1203   0.1138   0.1231   0.1133
## Detection Prevalence   0.2935   0.1805   0.1969   0.1917   0.1374
## Balanced Accuracy      0.8594   0.7736   0.7761   0.8346   0.7935
```


##Naive bayes
Third,The prediction Model was generated for training set using naive bayes algorithm. To using the naive bayes, we have to install 'kalR' packages.


```r
library(kalR)
set.seed(12345)
Cvctrl<-trainControl(method="cv", number=5, allowParallel=T, verbose=T)
NBmodFit <- train(classe~ .,data=Training,method="nb",prox=T,                                   trControl=Cvctrl,verbose=F)
```

###How accuracy model are?
####The result is not shown
confusionMatrix() function shows that how prediction model is accuracy.
By using 53 predictors for five classes using cross-validation at a 5-fold an accuracy of 75.4% with a 95% CI [0.7451-0.7643] was achieved accompanied by a Kappa value of 0.6915.


```r
NBpredict <- predict(NBmodFit, newdata=Testing)
confusionMatrix(NBpredict,Testing$classe)
```

##Conclusion 
In conclusion, RandomForest algorithm has the most accurative rate comparing with other two models, therfore, It is the best choice that applying RF model to final Testing set. 

##Testing set preprocessing
After buiding prediction model, we have to apply prediction model to test set(not a validation set).  

```r
Finaltesting<-read.csv("pml-testing.csv",na.strings=c("NA","#DIV/0!",""))
isNA <- apply(Finaltesting, 2, function(x) { sum(is.na(x)) })
Finaltesting <- subset(Finaltesting[, which(isNA == 0)], 
                    select=-c(X, user_name, new_window, num_window,                                    raw_timestamp_part_1,
                      raw_timestamp_part_2, cvtd_timestamp))

Finalprediction <- predict(modFit, newdata=Finaltesting)
Finaltesting$classe <- Finalprediction
Finaltesting$classe
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

##Submission
The below script was used to obtain 20 text files which have the target value predicted from prediction model. Each text files will be submitted to submission assignment.

```r
answers = Finaltesting$classe

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(answers)
```





