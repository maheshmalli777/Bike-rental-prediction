
# loading the libraries

x = c("tidyverse","ggplot2", "GGally", "plotly","magrittr","dplyr", "usdm", "xgboost","rattle", "Matrix","corrgram", "caret", "randomForest", "C50", "dummies", "e1071", "Information",
      "rpart", "gbm",'DataCombine')

setwd("C:/Users/headway/Desktop/m/Bike_rental_project")

# loading data
df_bike = read.csv("day.csv",header = T)
head(df_bike)

# Creating new dataset excluding casual and registered variables
df_bike = subset(df_bike,select=-c(casual,registered))
head(df_bike)

################################## Missing Values Analysis ###############################################

missing_val = data.frame(apply(df_bike,2,function(x){sum(is.na(x))}))
missing_val$Columns = row.names(missing_val)
names(missing_val)[1] = "Missing_percentage"
missing_val$Missing_percentage = (missing_val$Missing_percentage/nrow(df_bike)) * 100
missing_val = missing_val[order(-missing_val$Missing_percentage),]
row.names(missing_val) = NULL
missing_val = missing_val[,c(2,1)]
write.csv(missing_val, "R_Missing_perc.csv",row.names = F)
print(missing_val)

### Observation : No Missing value

########################################### Exploring the data ##########################################

# Structure of Dataset
str(df_bike)

# Shape of Dataset
dim(df_bike)

# Display Top records of dataset
head(df_bike)

################# Converting appropriate required Datatypes ####################

#Converting colmnns into categorical factors as they contain unqiue values

category_column_names = c('season','yr','mnth','holiday','weekday','weathersit', 'workingday')

for(i in category_column_names){
  df_bike[,i] = as.factor(df_bike[,i])
}

#Converting date into datetime format
df_bike$dteday = as.Date(df_bike$dteday ,"%d%m%Y")

#Converting rest variables into Numeric for standardization
for(i in 1:ncol(df_bike)){
  if(class(df_bike[,i]) == 'integer'){
    df_bike[,i] = as.numeric(df_bike[,i])
  }
}

########################################### Distribution Analysis ##########################################

#Distribution of Response Variable 'cnt'
library(magrittr)
library(dplyr)
library(plotly)
fit = density(df_bike$cnt)
plot_ly(x = df_bike$cnt, type = "histogram",name = "Count Distribution") %>%
  add_trace(x = fit$x,y = fit$y,type = "scatter", mode = "lines",fill = "tozeroy",yaxis = "y2",name = "Density") %>% 
  layout(yaxis2 = list(overlaying = "y",side = "right"))

#Bike Rental Count on Monthly Basis
df = aggregate(cnt ~ mnth,df_bike,sum)
plot_ly(x = ~mnth,y = ~cnt ,data = df, type = "bar",text = ~cnt ,marker = list(color = 'rgb(158,202,225)',line = list(color = 'rgb(8,48,107)',width = 1.5)))

#Bike Rental Count on Seasonal Basis
df = aggregate(cnt ~ season,df_bike,sum)
plot_ly(df,x = ~season, y = ~cnt,type = "scatter", mode = "lines+markers")

#Bike Rental Count on basis of Weather Type
df = aggregate(cnt ~ weathersit,df_bike, sum)
plot_ly(df, x = ~weathersit,y = ~cnt,type = "scatter",mode = "lines+markers")

#Bike Rental Count on basis of Day
df = aggregate(cnt ~ weekday,df_bike, sum)
plot_ly(df, x = ~weekday,y = ~cnt ,type = "box") %>%
  add_trace( x = ~weekday,y = ~cnt,type="scatter",mode = "lines+markers")

########################################### Outlier Analysis ##########################################

#boxplot for the cnt_outliers 

par(mfrow=c(1, 1))                   
#Here divide graph area in 1 columns and 1 rows 
boxplot(df_bike$cnt,main='Total_count',sub=paste(boxplot.stats(df_bike$cnt)$out))

#box plots for outliers 
par(mfrow=c(4,4)) 
#Box plot for the temp_outliers 
boxplot(df_bike$temp,main="Temp",sub=paste(boxplot.stats(df_bike$temp)$out))
#Box plot for the hum_outliers
boxplot(df_bike$hum,main="Humidity",sub=paste(boxplot.stats(df_bike$hum)$out)) 
#Box plot for the windspeed_outliers 
boxplot(df_bike$windspeed,main="Windspeed",sub=paste(boxplot.stats(df_bike$windspeed)$out))

#Replacing and imputate the outliers 
 
library(DMwR) 
#create subset for windspeed and hum variable
wind_hum = subset(df_bike,select=c('windspeed','hum'))
#column names of the wind_hum
cnames = colnames(wind_hum) 
for(i in cnames){ 
  val=wind_hum[,i][wind_hum[,i] %in% boxplot.stats(wind_hum[,i])$out]
  wind_hum[,i][wind_hum[,i] %in% val]= NA                                 # Replacing the outliers with NA
}
#Imputating the missing values using mean imputation method 
wind_hum$windspeed[is.na(wind_hum$windspeed)] = mean(wind_hum$windspeed,na.rm=T)
wind_hum$hum[is.na(wind_hum$hum)] = mean(wind_hum$hum,na.rm=T)

#Remove the windspeed and humidity variable in order to replace imputated data 
new_df = subset(df_bike,select=-c(windspeed,hum))
#Combined new_df and wind_hum data frames 
df_bike = cbind(new_df,wind_hum) 
head(df_bike,5)

########################################### Feature Selection  ##########################################

#Normality test and Correlation matrix

#Quintle-Quintle normal plot 
normal_plot = qqnorm(df_bike$cnt) 
print(normal_plot)
#Quintle-Quintle line 
qqline(df_bike$cnt)

## Correlation Plot 
library(corrgram) 
library(seriation)
#Correlation plot 
corrgram(df_bike[,8:14],order=F,upper.panel=panel.pie,text.panel=panel.txt,main='Correlation_Plot')

########################################### Model Development ##########################################

#load the purrr library for functions and vectors 
library(purrr) 
#Split the dataset based on simple random resampling 
train_index = sample(1:nrow(df_bike),0.7*nrow(df_bike)) 
data_train = df_bike[train_index,] 
data_test = df_bike[-train_index,] 
dim(data_train) 
dim(data_test)

#Create a new subset for train attributes 
train = subset(data_train,select=c('season','yr','mnth','holiday','weekday','workingday','weathersit','temp','hum','windspeed','cnt'))
#Create a new subset for test attributes 
test = subset(data_test,select=c('season','yr','mnth','holiday','weekday', 'workingday','weathersit','temp','hum','windspeed','cnt'))

#create a new subset for train categorical attributes
train_cat_attributes = subset(train,select=c('season','holiday','workingday', 'weathersit','yr')) 
#create a new subset for test categorical attributes
test_cat_attributes = subset(test,select=c('season','holiday','workingday', 'weathersit','yr'))
#create a new subset for train numerical attributes 
train_num_attributes = subset(train,select=c('weekday','mnth','temp','hum','windspeed','cnt')) 
#create a new subset for test numerical attributes
test_num_attributes = subset(test,select=c('weekday','mnth','temp','hum','windspeed','cnt'))

#load the caret library 
library(caret) 
#other variables along with target variable to get dummy variables
othervars = c('mnth','weekday','temp','hum','windspeed','cnt')

#Categorical variables
vars = setdiff(colnames(train),c(train$cnt,othervars)) 
#formula pass through encoder to get dummy variables 
f = paste('~', paste(vars,collapse =' + '))
#encoder is encoded the categorical variables to numeric 
encoder = dummyVars(as.formula(f),train)
#Predicting the encode attributes 
encode_attributes = predict(encoder,train)
#Binding the train_num_attributes and encode_attributes 
train_encoded_attributes = cbind(train_num_attributes,encode_attributes)
head(train_encoded_attributes,5)

#Categorical variables
vars = setdiff(colnames(test),c(test$cnt,othervars))
#formula pass through encoder to get dummy variables
f = paste('~',paste(vars,collapse='+'))
#Encoder is encoded the categorical variables to numeric 
encoder = dummyVars(as.formula(f),test) 
#Predicting the encoder attributes
encode_attributes = predict(encoder,test) 
#Binding the test_num_attributes and encode_attributes 
test_encoded_attributes = cbind(test_num_attributes,encode_attributes)
head(test_encoded_attributes,5)

########################################### Linear Regression model ##########################################

#training the lr_model
lr_model = lm(train_encoded_attributes$cnt~.,train_encoded_attributes[,-c(6)]) 

#Cross validation prediction
#Set seed to reproduce results of random sampling
set.seed(623) 
#Cross validation resampling method 
train.control = trainControl(method='CV',number=3)
#Cross validation prediction 
CV_predict = train(cnt~.,data=train_encoded_attributes,method='lm',trControl=train.control) 
#Summary of cross validation prediction 
summary(CV_predict)

#Cross validation prediction plot 
residuals = resid(CV_predict) 
y_train = train_encoded_attributes$cnt
plot(y_train,residuals,ylab =('Residuals'),xlab =('Observed'),main =('CV plot')) 
abline(0,0)

#predict the lr_model
lm_predict = predict(lr_model,test_encoded_attributes[,-c(6)])
head(lm_predict,5)

#Root mean squared error 
rmse = RMSE(lm_predict,test_encoded_attributes$cnt) 
print(rmse) 
#Mean squared error 
mae = MAE(lm_predict,test_encoded_attributes$cnt) 
print(mae)

#Residual plot
y_test = test_encoded_attributes$cnt
residuals = y_test-lm_predict 
plot(y_test,residuals,xlab ='Observed',ylab ='Residuals',main ='Residual plot') 
abline(0,0)

########################################### Decision tree model ##########################################

library(rpart)

#rpart.control to contro the performance of model
rpart.control<-rpart.control(minbucket = 2,cp = 0.01,maxcompete = 3,maxsurrogate = 4, usesurrogate = 2,xval = 3,surrogatestyle = 0,maxdepth =10)

#training the dtr model 
dt_model = rpart(train_encoded_attributes$cnt~.,data=train_encoded_attributes[,-c(6)],control=rpart.control,method='anova',cp=0.01)

#load the rpart.plot for plot the learned dtr model 
library(rpart.plot) 
rpart.plot(dt_model,box.palette="RdBu",shadow.col="gray", nn=TRUE,roundint=FALSE)

#cross validation resampling method 
train.control = trainControl(method='CV',number=3) 
#cross validation pred 
dt_CV_predict = train(cnt~.,data=train_encoded_attributes,method='rpart',trControl=train.control)

#Cross validation prediction plot
residuals = resid(dt_CV_predict) 
plot(y_train,residuals,xlab='Observed',ylab='Residuals',main='Cross validation plot') 
abline(0,0)

#predict the trained model 
dt_predict = predict(dt_model,test_encoded_attributes[,-c(6)])
head(dt_predict,5)

#Root mean squared error 
rmse = RMSE(y_test,dt_predict) 
print(rmse) 
#Mean absolute error 
mae = MAE(y_test,dt_predict) 
print(mae)

#Residual plot 
residuals = y_test-dt_predict
plot(y_test,residuals,xlab='Observed',ylab='Residuals',main='Residual plot')
abline(0,0)

########################################### Random Forest model ##########################################

library(randomForest)

#training the rf model 
rf_model = randomForest(cnt~.,train_encoded_attributes,importance=TRUE,ntree=200)

library(ranger) 
#Cross validation resampling method 
train.control = trainControl(method='CV',number=3) 
#Cross validation prediction
rf_CV_predict = train(cnt~.,train_encoded_attributes,method='ranger',trControl=train.control)

#Cross validation prediction plot 
residuals = resid(rf_CV_predict) 
plot(y_train,residuals,xlab='Observed',ylab='Residuals',main='Cross validation prediction plot') 
abline(0,0)

#Predicting the model 
rf_predict = predict(rf_model,test_encoded_attributes[,-c(6)])
head(rf_predict,5)

#Root mean squared error 
rmse = RMSE(y_test,rf_predict) 
print(rmse)
#Mean absolute error
mae = MAE(y_test,rf_predict) 
print(mae)

#Residual plot 
residuals = y_test-rf_predict 
plot(y_test,residuals,xlab='Observed',ylab='Residuals',main='Residual plot') 
abline(0,0)

########################################### Saving the model ##########################################
Bike_predictions=data.frame(y_test,rf_predict)
write.csv(Bike_predictions,'Bike_Renting_R.CSV',row.names=F)
