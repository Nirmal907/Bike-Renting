rm(list=(ls()))
getwd()

#Load the data
data=read.csv("E:/Edwisor/Projects/Bike Rent/day.csv",header=T)

# installing packages

x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "C50","xgboost", "e1071","gridExtra",
      "MASS",'DataCombine', "gbm","dplyr","Matrix","dummies")

lapply(x, require, character.only = TRUE)
rm(x)
str(data)

# Missing Value
sapply(data, function(x) sum(is.na(x)))

#Copy of a a data
data_copy=data

# Removing Features "instant" , "dteday" ,"registered" and "casual" from the dataset. These features does have any relevance.
data_copy=subset(data_copy,select=-c(instant,dteday,registered,casual))       
dim(data_copy)

# Changing the datatype of target variable from integer to numeric
data_copy$cnt=as.numeric(data_copy$cnt)


#Changing the datatype to factor and assigning the labels
for(i in 1:ncol(data_copy)){
  if(class(data_copy[,i]) == 'integer'){
    data_copy[,i]=as.factor(data_copy[,i])
    data_copy[,i]= factor(data_copy[,i],labels = 1:length(levels(factor(data_copy[,i]))))
  }
}

str(data_copy)


#Getting all the numeric columns from the data frame
numeric_index = sapply(data_copy,is.numeric)
numeric_data=data_copy[,numeric_index]
cnames = colnames(numeric_data)

cnames = cnames[-5]
cnames

#Creating box plot of the numeric data for outlier analysis
for ( i in 1:length(cnames))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = "cnt"), data = subset(data_copy))+ 
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="blue", fill = "grey" ,outlier.shape=1,
                        outlier.size=3, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(x="Count")+
           ggtitle(paste(cnames[i])))+scale_x_discrete(breaks = NULL)
}



#Plotting Box Plot on the Plot window
gridExtra::grid.arrange(gn1,gn2,gn3,gn4,ncol=4)

#Creating the histogram of the features having outliers plot
hist(data_copy$windspeed)
hist(data_copy$hum)


#Finding all the outliers in the data set
for(i in cnames){
  val = data_copy[,i][data_copy[,i] %in% boxplot.stats(data_copy[,i])$out]
  data_copy[,i][data_copy[,i] %in% val] = NA
}

#Calculating Missing values in data frame(train_data)
missing_val_removed=data.frame(apply(data_copy,2,function(x){sum(is.na(x))}))

#Imputing NA values with mean Method
data_copy$hum[is.na(data_copy$hum)] = mean(data_copy$hum, na.rm = T)
data_copy$windspeed[is.na(data_copy$windspeed)] = mean(data_copy$windspeed, na.rm = T)



#Histogram of features after removing the outliers
hist(data_copy$windspeed)
hist(data_copy$hum)



#Feature Selection# 
#Plotting Correlation plot of the numeric data
corrgram(data_copy[,numeric_index], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")


#Checking the correlation among all the features using variance inflation factor.

data_copy_2= data_copy
for(i in 1:ncol(data_copy_2)){
  if(class(data_copy_2[,i]) == 'factor'){
    data_copy_2[,i]=as.numeric(data_copy_2[,i])
  }
}
#Multicollinearity test
install.packages("usdm")
library(usdm)

vif(data_copy_2[,-12])
vifcor(data_copy_2[,-12],th=0.9)


#Feature Selection

data_copy3 = subset(data_copy,select =-c(atemp))
str(data_copy3)

#Modeling#

#Spliting the data into train and test data
set.seed(1234)
train.index = sample(1:nrow(data_copy3),0.8*nrow(data_copy3))
train_data = data_copy3[train.index,]
test_data = data_copy3[-train.index,]

str(train_data)


#Defining function which will we used to find the accuracy of the model
## Mean Absolute Percentage Error
MAPE = function(y, yhat){
  mean(abs((y - yhat)/y))*100
}

## Mean Absolute Error
MAE = function(y,f){
  mean(abs(y-f))
}

## Root Mean Square Error
RMSE = function(y,f){
  sqrt(mean((y-f)^2))
}


## Accuracy
Acc = function(test_data_true, predicted_values){
  mean_abs_error = format(round(MAE(test_data_true,predicted_values),2),nsmall = 2)
  root_mean_sq_er = format(round(RMSE(test_data_true,predicted_values),2),nsmall = 2)
  Error = format(round(MAPE(test_data_true,predicted_values),2),nsmall = 2)
  Accuracy = 100 - as.numeric(Error)
  print(paste0("Mean Absolute Error : ", mean_abs_error))
  print(paste0("Mean Absolute Percentage Error : " , Error))
  print(paste0("Root Mean Square Error : ", root_mean_sq_er))
  print(paste0("Accuracy : ", Accuracy))
}



#Decision Tree# 
install.packages("rpart.plot")
library(rpart.plot)
library(rpart)

##Rpart for regression
dt_model = rpart(cnt~.,data=train_data,method = 'anova')
rpart.plot(dt_model)
rpart.rules(dt_model)
predict_dt = predict(dt_model,test_data[,-11])
Acc(test_data[,11],predict_dt)


## Error Rate 21.98
## Accuracy 78.02


#Linear Regression
lm_model= lm(cnt~.,data= train_data)
summary(lm_model)
predictions_LR = predict(lm_model,test_data[,-11])
Acc(test_data[,11], predictions_LR)

## Error Rate 20.71
## Accuracy 79.29

#Random Forest Aglorithm
RF_model = randomForest(cnt ~ ., train_data, importance = TRUE, ntree = 300)
RF_Predictions = predict(RF_model, test_data[,-11])

Acc(test_data[,11], RF_Predictions)

## Error 19.91
## Accuracy 80.09


##Gradient Boosting Algorithm##
## creating Spare Matrix which converts Categorical Variables to dummy variables
trainm = sparse.model.matrix(cnt~.-1,data = train_data)
train_label <- train_data[,"cnt"]
train_matrix = xgb.DMatrix(data = as.matrix(trainm),label = train_label)

testm = sparse.model.matrix(cnt~.-1,data = test_data)
test_label = test_data[,"cnt"]
test_matrix = xgb.DMatrix(data = as.matrix(testm), label = test_label)



#Defining Parameters for Xgboost
params <- list(booster = "gbtree", objective = "reg:linear", eta=0.2, gamma=0, max_depth=6, 
               min_child_weight=1, 
               subsample=1, colsample_bytree=1)

## Cross Validation - for finding the minimum number of rounds required to find the best accuracy

xgbcv <- xgb.cv( params = params, data = train_matrix, nrounds = 100, nfold = 5, showsd = F, stratified = T, 
                 maximize = F)

## Creating watchlist
watchlist = list(train = train_matrix, test = test_matrix)

## Apply XGBoost Algorithm for train data
xgb1 <- xgb.train (params = params, data = train_matrix, nrounds = 43, watchlist = watchlist, 
                   print_every_n = 10, early_stop_round = 10, maximize = F , eval_metric = "rmse")

## Predicting the values of test data using training data
xgb_predict = predict(xgb1,test_matrix)
Acc(test_data[,11],xgb_predict)

##Error 16.06
##Acuracy 83.94

