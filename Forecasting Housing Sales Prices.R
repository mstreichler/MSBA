update.packages()

install.packages("ggplot2")
install.packages("plyr")
install.packages("caret")
install.packages("moments")
install.packages("glmnet")
install.packages("elasticnet")
install.packages("knitr")
install.packages("SpatialNP")
install.packages("MASS")
install.packages("dplyr")
install.packages("randomForest")
install.packages("pls")
install.packages("scales")
install.packages("ggrepel")

library(ggplot2)
library(plyr)
library(caret)
library(moments)
library(glmnet)
library(elasticnet)
library(knitr)
library(kknn)
library(corrplot)
library(SpatialNP)
library(MASS)
library(dplyr)
library(randomForest)
library(pls)
library(scales)
library(ggrepel)

#Run everything above this line, one time. For multiple iterations of this code, run everything below this line.

rm(list=ls())
#We are provided with a validation set approach, where a set% of the data is training and test.
#We will remove Salesprice from training df to allow for KFold cross validation
train = read.csv("train.csv", stringsAsFactors = FALSE)
test = read.csv("test.csv", stringsAsFactors = FALSE)

#Skewness plot
ggplot(data=train[!is.na(train$SalePrice),], aes(x=SalePrice)) + geom_histogram(fill="blue", binwidth = 10000) + scale_x_continuous(breaks= seq(0, 800000, by=100000), labels = comma)

#Important Variable Box Plots and Scatter
ggplot(data=train[!is.na(train$SalePrice),], aes(x=factor(OverallQual), y=SalePrice))+
  geom_boxplot(col='blue') + labs(x='Overall Quality') +
  scale_y_continuous(breaks= seq(0, 800000, by=100000))

ggplot(data=train[!is.na(train$SalePrice),], aes(x=factor(YearBuilt), y=SalePrice))+
  geom_boxplot(col='blue') + labs(x='YearBuilt') +
  scale_y_continuous(breaks= seq(0, 800000, by=100000))+theme(axis.text.x = element_text(angle = -90))

ggplot(data=train[!is.na(train$SalePrice),], aes(x=factor(GarageCars), y=SalePrice))+
  geom_boxplot(col='blue') + labs(x='GarageCars') +
  scale_y_continuous(breaks= seq(0, 800000, by=100000))

ggplot(data=train[!is.na(train$SalePrice),], aes(x=GrLivArea, y=SalePrice))+
  geom_point(col='blue') + geom_smooth(method = "lm", se=FALSE, color="black", aes(group=1)) +
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) +
  geom_text_repel(aes(label = ifelse(train$GrLivArea[!is.na(train$SalePrice)]>4500, rownames(train), '')))

ggplot(data=train[!is.na(train$SalePrice),], aes(x=LotArea, y=SalePrice))+
  geom_point(col='blue') + geom_smooth(method = "lm", se=FALSE, color="black", aes(group=1)) +
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma)

#Correlatin Matrix for train dataset
numericVars = which(sapply(train, is.numeric)) #index vector numeric variables
numericVarNames = names(numericVars) #saving names vector for use later on
train_numVar <- train[, numericVars]
cor_numVar <- cor(train_numVar, use="pairwise.complete.obs") #correlations of all numeric variables
cor_sorted <- as.matrix(sort(cor_numVar[,'SalePrice'], decreasing = TRUE))
CorHigh <- names(which(apply(cor_sorted, 1, function(x) abs(x)>0.5)))
cor_numVar <- cor_numVar[CorHigh, CorHigh]
corrplot.mixed(cor_numVar, tl.col="black", tl.pos = "lt")




#See if normally distributed visually before log
qqnorm(train$SalePrice)
qqline(train$SalePrice)


#Separate salesprice into its own df from train df, before performing calculations and apply log.
train$SalePrice = log(train$SalePrice + 1)
y = train$SalePrice



#See if normally distributed visually after log
qqnorm(y)
qqline(y)



#Combine train and test into data df. This allows for pre-processing to be applied once rather than on each matrix. Exclude salesprice when combining
#We also dropped "ID" column here
data = rbind(select(train,MSSubClass:SaleCondition),
                  select(test,MSSubClass:SaleCondition))


#Find numeric and categorical feature variables
#We will use "sapply" here which is a variation of "apply". There are 3 variations of apply, "sapply","lapply","vapply".
#Feature class is utilizing "sapply" which is converting the data df into a vector which is called "x". 
#The "name" parameter is keeping the column names of data df. "function(x)" is allowing us to utilize a custom function in "sapply", 
#normally it has built in functions like "sum".
#The "class" parameter is replacing the values in the vector "x" with the type, in this case, "integer", and "character".
#So we now have a vector "x" with the same column headers as data df, but the 1 row of values have been replaced with "character" or "integer". 
feature_classes = sapply(names(data),function(x){class(data[[x]])})
#Here we are taking the column names with "names" and keeping only the columns that do not equal "character" type.
numeric_feats = names(feature_classes[feature_classes != "character"])
#Here we are keeping columns that do equal "character" type.
categorical_feats = names(feature_classes[feature_classes == "character"])



#Remove Skewnewss
#for numeric feature with excessive skewness, perform log transformation
#determine skew for each numeric feature
skewed_feats = sapply(numeric_feats,function(x){skewness(data[[x]],na.rm=TRUE)})

#keep only features that exceed a threshold for skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]

#transform excessively skewed features with log(x + 1)
for(x in names(skewed_feats)) {
  data[[x]] = log(data[[x]] + 1)
}




#One hot encode/ make dummy variables
#"dummyVars" collects all the information needed to produce a full set of dummy variables. 
#"~." will look at the variables not explicitly called, and replace the "." with those missing variables. This is all the variables in this case.
#We choose the "categorical_feats" as the vector to one hot encode.
dummies = dummyVars(~.,data[categorical_feats])
#Here we are utilizing the data gathered in "dummies" to one hot encode with the "predict" function which produces a df
categorical_1_hot = predict(dummies,data[categorical_feats])
#This will convert "NA" to 0 which avoids the NA error then
categorical_1_hot[is.na(categorical_1_hot)] = 0  
#Place all numeric variables into numeric df
numeric_df = data[numeric_feats]



#remove variables with high % of NA
numeric_df = numeric_df[, which(colMeans(is.na(numeric_df)) < 0.1)]

#combine numeric_df amd categprocal_1_hot into data
data = cbind(numeric_df,categorical_1_hot)


#Split the data df back into train and test df
train = data[1:nrow(train),]
test = data[(nrow(train)+1):nrow(data),]

#We make numeric variable df for the train and test df
feature_classes_train = sapply(names(train),function(x){class(train[[x]])})
feature_classes_test = sapply(names(test),function(x){class(test[[x]])})
numeric_feats_train = names(feature_classes_train[feature_classes_train != "character"])
numeric_feats_test = names(feature_classes_test[feature_classes_test != "character"])

#Impute Numeric feature variables from before one hot encoding, because we believe dummy variables inherently have a high multicolinearty.
#"x" is the numeric vector. "na.rm=TRUE" determines if NA should be removed, in this case, yes.
#Before removing the na.rm, we instead add a line "is.na" to check if the line is NA, then replace it with the mean_value
for (x in numeric_feats_train) {
  mean_value = mean(train[[x]],na.rm = TRUE)
  train[[x]][is.na(train[[x]])] = mean_value
}

for (x in numeric_feats_test) {
  mean_value = mean(test[[x]],na.rm = TRUE)
  test[[x]][is.na(test[[x]])] = mean_value
}





#test = test[,-aux] #and remove the columns
test_logic_names = colSums(abs(test), na.rm = TRUE) > 0
test_logic_names_index = which(!test_logic_names)
test=test[colSums(abs(test), na.rm = TRUE) > 0]
train=train[, c(-test_logic_names_index)]

#Scale data in train and test, this will change dummy variables to no longer be just 0 and 1, Professor said its fine to do that.
train = scale(train)
test = scale(test)


#Spatial Sign(should be last data-preprocessing method) accounts for outliers (https://cran.r-project.org/web/packages/SpatialNP/SpatialNP.pdf)
#train_sp=spatial.signs(train, center=TRUE, shape=TRUE)


#Convert test and train matrix to data frames
train=as.data.frame(train)
test=as.data.frame(test)


##########KNN with Kfold
#Collects the total amount of rows in the combined dataset called 'data'
n = dim(train)[1]

train$SalePrice <- y
#Define the number of folds
kcv = 10 #In this case, 10-fold cross validation
#This k has nothing to do with the k from knn

#Size of the fold (which is the number of elements in the test matrix)
n0 = round(n/kcv, #Number of observations in the fold
           0) #Rounding with 0 decimals

#Number of neighbors for different models
kk <- 1:100
# kk <- 1

#MSE matrix
#10 rows x 100 columns
out_MSE = matrix(0, #matrix filled with zeroes
                 nrow = kcv, #number of rows
                 ncol = length(kk)) #number of columns

#Vector of indices that have already been used inside the for
used = NULL

#The set of indices not used (will be updated removing the used)
set = 1:n


for(j in 1:kcv){
  
  if(n0<length(set)){ #If the set of 'not used' is > than the size of the fold
    val = sample(set, size = n0) #then sample indices from the set
  }
  
  if(n0>=length(set)){ #If the set of 'not used' is <= than the size of the fold
    val=set #then use all of the remaining indices as the sample
  }
  
  #Create the train and test matrices
  train_t = train[-val,] #Every observation except the ones whose indices were sampled
  test_t = train[val,] #The observations whose indices sampled
  
  for(i in kk){
    
    #The current model
    near = kknn(SalePrice ~., #The formula
                train = train_t, #The train matrix/df
                test = test_t, #The test matrix/df
                k=i, #Number of neighbors
                kernel = "rectangular") #Type of kernel (see help for more)
    
    #Calculating the MSE of current model
    aux = mean((test_t$SalePrice-near$fitted.values)^2)
    
    #Store the current MSE
    out_MSE[j,i] = aux
  }
  
  #The union of the indices used currently and previously
  used = union(used,val)
  
  #The set of indices not used is updated
  set = (1:n)[-used]
  
  #Printing on the console the information that you want
  #Useful to keep track of the progress of your loop
  cat(j,"folds out of",kcv,'\n')
}


#Calculate the mean of MSE for each k
mMSE = apply(out_MSE, #Receive a matrix
             2, #Takes its columns (it would take its rows if this argument was 1)
             mean) #And for each column, calculate the mean
#Complexity x RMSE graph
plot(log(1/kk),sqrt(mMSE),
     xlab="Complexity (log(1/k))",
     ylab="out-of-sample RMSE",
     col=4, #Color of line
     lwd=2, #Line width
     type="l", #Type of graph = line
     cex.lab=1.2, #Size of labs
     main=paste("kfold(",kcv,")")) #Title of the graph
#Find the index of the minimum value of mMSE
best = which.min(mMSE)
#Including text at specific coordinates of the graph
text(log(1/kk[best]),sqrt(mMSE[best])+0.01, #Coordinates
     paste("k=",kk[best]), #The actual text
     col=2, #Color of the text
     cex=1.2) #Size of the text
text(log(1/2),sqrt(mMSE[2])+0.01,
     paste("k=",2),
     col=2,
     cex=1.2)
text(log(1/100)+0.4,sqrt(mMSE[100]),
     paste("k=",100),
     col=2,
     cex=1.2)
#Find the index of the minimum value of mMSE
best = which.min(mMSE)
optimal = kk[best]      #Optimal KNN = 9
minRMSE = sqrt(mMSE[best])
minRMSE                #Lowest RMSE = 0.1987424
near = kknn(SalePrice ~., #The formula
            train = train, #The train matrix/df
            test = test, #The test matrix/df
            k=optimal, #Number of neighbors
            kernel = "rectangular")
# make create submission file
knn_predict = exp(predict(near,newdata=test)) - 1
# construct data frame for solution
solution = data.frame(Id=as.integer(rownames(test)),SalePrice=knn_predict)
write.csv(solution,"kfold.csv",row.names=FALSE)




###########Forward Stepwise Regression
forwards = lm(train$SalePrice ~., data = train)
step.model <- stepAIC(forwards, direction = "forward", trace = FALSE)
# names of the coefficients stepwise model choose
# names(step.model$coefficients)
# final model
step.model$anova
# plot
par(mfrow=c(2,2))
plot(step.model)
stepwise.final = exp(predict.lm(step.model, test)) - 1
title = c("SalePrice")
names(title)= c("Id")
write.table(title,file="lm.csv",col.names =F, sep = ",")
write.table(stepwise.final,file="lm.csv",col.names =F, sep = ",", append = T)


##########Lasso

rm(list=ls())
setwd("C:/Users/rehan/OneDrive/School work/UT Austin/Summer 2020/STA S380/Part 1/Project/Project Files")
#We are provided with a validation set approach, where a set% of the data is training and test.
#We will remove Salesprice from training df to allow for KFold cross validation
train = read.csv("train.csv", stringsAsFactors = FALSE)
test = read.csv("test.csv", stringsAsFactors = FALSE)

#Separate salesprice into its own df from train df, before performing calculations and apply log.
train$SalePrice = log(train$SalePrice + 1)
y = train$SalePrice

#Combine train and test into data df. This allows for pre-processing to be applied once rather than on each matrix. Exclude salesprice when combining
#We also dropped "ID" column here
data = rbind(select(train,MSSubClass:SaleCondition),
             select(test,MSSubClass:SaleCondition))


#Find numeric and categorical feature variables
#We will use "sapply" here which is a variation of "apply". There are 3 variations of apply, "sapply","lapply","vapply".
#Feature class is utilizing "sapply" which is converting the data df into a vector which is called "x". 
#The "name" parameter is keeping the column names of data df. "function(x)" is allowing us to utilize a custom function in "sapply", 
#normally it has built in functions like "sum".
#The "class" parameter is replacing the values in the vector "x" with the type, in this case, "integer", and "character".
#So we now have a vector "x" with the same column headers as data df, but the 1 row of values have been replaced with "character" or "integer". 
feature_classes = sapply(names(data),function(x){class(data[[x]])})
#Here we are taking the column names with "names" and keeping only the columns that do not equal "character" type.
numeric_feats = names(feature_classes[feature_classes != "character"])
#Here we are keeping columns that do equal "character" type.
categorical_feats = names(feature_classes[feature_classes == "character"])



#Remove Skewnewss
#for numeric feature with excessive skewness, perform log transformation
#determine skew for each numeric feature
skewed_feats = sapply(numeric_feats,function(x){skewness(data[[x]],na.rm=TRUE)})

#keep only features that exceed a threshold for skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]

#transform excessively skewed features with log(x + 1)
for(x in names(skewed_feats)) {
  data[[x]] = log(data[[x]] + 1)
}




#One hot encode/ make dummy variables
#"dummyVars" collects all the information needed to produce a full set of dummy variables. 
#"~." will look at the variables not explicitly called, and replace the "." with those missing variables. This is all the variables in this case.
#We choose the "categorical_feats" as the vector to one hot encode.
dummies = dummyVars(~.,data[categorical_feats])
#Here we are utilizing the data gathered in "dummies" to one hot encode with the "predict" function which produces a df
categorical_1_hot = predict(dummies,data[categorical_feats])
#This will convert "NA" to 0 which avoids the NA error then
categorical_1_hot[is.na(categorical_1_hot)] = 0  
#Place all numeric variables into numeric df
numeric_df = data[numeric_feats]



#remove variables with high % of NA
numeric_df = numeric_df[, which(colMeans(is.na(numeric_df)) < 0.1)]


#combine numeric_df amd categprocal_1_hot into data
data = cbind(numeric_df,categorical_1_hot)


#Split the data df back into train and test df
train = data[1:nrow(train),]
test = data[(nrow(train)+1):nrow(data),]

#We make numeric variable df for the train and test df
feature_classes_train = sapply(names(train),function(x){class(train[[x]])})
feature_classes_test = sapply(names(test),function(x){class(test[[x]])})
numeric_feats_train = names(feature_classes_train[feature_classes_train != "character"])
numeric_feats_test = names(feature_classes_test[feature_classes_test != "character"])

#Impute Numeric feature variables from before one hot encoding, because we believe dummy variables inherently have a high multicolinearty.
#"x" is the numeric vector. "na.rm=TRUE" determines if NA should be removed, in this case, yes.
#Before removing the na.rm, we instead add a line "is.na" to check if the line is NA, then replace it with the mean_value
for (x in numeric_feats_train) {
  mean_value = mean(train[[x]],na.rm = TRUE)
  train[[x]][is.na(train[[x]])] = mean_value
}

for (x in numeric_feats_test) {
  mean_value = mean(test[[x]],na.rm = TRUE)
  test[[x]][is.na(test[[x]])] = mean_value
}



#test = test[,-aux] #and remove the columns
test_logic_names = colSums(abs(test), na.rm = TRUE) > 0
test_logic_names_index = which(!test_logic_names)
test=test[colSums(abs(test), na.rm = TRUE) > 0]
train=train[, c(-test_logic_names_index)]

#Scale data in train and test, this will change dummy variables to no longer be just 0 and 1, Professor said its fine to do that.
train = scale(train)
test = scale(test)


#Spatial Sign(should be last data-preprocessing method) accounts for outliers (https://cran.r-project.org/web/packages/SpatialNP/SpatialNP.pdf)
#train_sp=spatial.signs(train, center=TRUE, shape=TRUE)


#Convert test and train matrix to data frames
train=as.data.frame(train)
test=as.data.frame(test)

# set up caret model training parameters
# model specific training parameter
CARET.TRAIN.CTRL = trainControl(method="repeatedcv",
                                 number=10,
                                 repeats=10,
                                 verboseIter=FALSE)


set.seed(123)  # for reproducibility
model_lasso = train(x=train,y=y,
                     method="glmnet",
                     metric="RMSE",
                     maximize=FALSE,
                     trControl=CARET.TRAIN.CTRL,
                     tuneGrid=expand.grid(alpha=1,  # Lasso regression
                                          lambda=c(1,0.1,0.05,0.01,seq(0.009,0.001,-0.001),
                                                   0.00075,0.0005,0.0001)))
model_lasso

mean(model_lasso$resample$RMSE)

# extract coefficients for the best performing model
coef = data.frame(coef.name = dimnames(coef(model_lasso$finalModel,s=model_lasso$bestTune$lambda))[[1]], 
                   coef.value = matrix(coef(model_lasso$finalModel,s=model_lasso$bestTune$lambda)))

# exclude the (Intercept) term
coef = coef[-1,]

# print summary of model results
picked_features = nrow(filter(coef,coef.value!=0))
not_picked_features = nrow(filter(coef,coef.value==0))

cat("Lasso picked",picked_features,"variables and eliminated the other",
    not_picked_features,"variables\n")

# sort coefficients in ascending order
coef = arrange(coef,-coef.value)

# extract the top 10 and bottom 10 features
imp_coef = rbind(head(coef,10),
                  tail(coef,10))

ggplot(imp_coef) +
  geom_bar(aes(x=reorder(coef.name,coef.value),y=coef.value),
           stat="identity") +
  ylim(-1.5,0.6) +
  coord_flip() +
  ggtitle("Coefficents in the Lasso Model") +
  theme(axis.title=element_blank())

ggplot(data=filter(model_lasso$result,RMSE<0.14)) +
  geom_line(aes(x=lambda,y=RMSE))

# make create submission file
preds = exp(predict(model_lasso,newdata=test)) - 1

# construct data frame for solution
solution = data.frame(Id=as.integer(rownames(test)),SalePrice=preds)
write.csv(solution,"lasso.csv",row.names=FALSE)





##########Random Forest

# reload the datas again so that the variables are not scaled and the categorical variables are just as they are
train_tree = read.csv("train.csv", stringsAsFactors = FALSE)
test_tree = read.csv("test.csv", stringsAsFactors = FALSE)

# separate salesprice in the training data set and make it the y variable
train_tree$SalePrice = log(train_tree$SalePrice + 1)
y_tree = train_tree$SalePrice
train_tree <- train_tree[,-81]

# combine the training data and test data to remove variables with too many NAs
data_tree = rbind(train_tree[,-1], test_tree[,-1])

# separate numerical and categorical variables
# only going to remove the columns with too many NAs for numerical ones
feature_classes_tree = sapply(names(data_tree),function(x){class(data_tree[[x]])})
numeric_feats_tree = names(feature_classes_tree[feature_classes_tree != "character"])
numeric_df_tree = data_tree[numeric_feats_tree]
numeric_df_tree = numeric_df_tree[, which(colMeans(is.na(numeric_df_tree)) < 0.1)]

# recombine numerical and categorical variables
categorical_feats_tree = names(feature_classes_tree[feature_classes_tree == "character"])
categorical_df_tree = data_tree[categorical_feats_tree]
data_tree = cbind(numeric_df_tree,categorical_df_tree)

# split the data df back into train and test df
train_tree = data_tree[1:nrow(train_tree),]
test_tree = data_tree[(nrow(train_tree)+1):nrow(data_tree),]

# we make numeric and categorical variable df for the train and test df
feat_classes_train_tree = sapply(names(train_tree),function(x){class(train_tree[[x]])})
feat_classes_test_tree = sapply(names(test_tree),function(x){class(test_tree[[x]])})

num_feats_train_tree = names(feat_classes_train_tree[feat_classes_train_tree != "character"])
num_feats_test_tree = names(feat_classes_test_tree[feat_classes_test_tree != "character"])
cat_feats_train_tree = names(feat_classes_train_tree[feat_classes_train_tree == "character"])
cat_feats_test_tree = names(feat_classes_test_tree[feat_classes_test_tree == "character"])

# impute na with mean value of the column for numerical variables
for (x in num_feats_train_tree) {
  mean_value = mean(train_tree[[x]],na.rm = TRUE)
  train_tree[[x]][is.na(train_tree[[x]])] = mean_value
}

for (x in num_feats_test_tree) {
  mean_value = mean(test_tree[[x]],na.rm = TRUE)
  test_tree[[x]][is.na(test_tree[[x]])] = mean_value
}


# impute na with "None" for categorical variables
for (x in cat_feats_train_tree) {
  train_tree[[x]][is.na(train_tree[[x]])] = "None"
}

for (x in cat_feats_test_tree) {
  test_tree[[x]][is.na(test_tree[[x]])] = "None"
}

# add back the salesprice column
train_tree["Salesprice"] <- y_tree

# apply k-fold and random forest
# define the number of folds
kcv = 10 #In this case, 10-fold cross validation

#Size of the fold (which is the number of elements in the test matrix)
n = dim(train_tree)[1]
n0 = round(n/kcv, 0) 

ntreev = c(10,200,500,1000) #Different numbers of trees
nset = length(ntreev)

out_MSE = matrix(0, #matrix filled with zeroes
                 nrow = kcv, #number of rows
                 ncol = length(ntreev)) #number of columns

#Vector of indices that have already been used inside the for
used = NULL

#The set of indices not used (will be updated removing the used)
set = 1:n

for(j in 1:kcv){
  
  if(n0<length(set)){ #If the set of 'not used' is > than the size of the fold
    val = sample(set, size = n0) #then sample indices from the set
  }
  
  if(n0>=length(set)){ #If the set of 'not used' is <= than the size of the fold
    val=set #then use all of the remaining indices as the sample
  }
  
  #Create the train and test matrices
  train_i = train_tree[-val,] #Every observation except the ones whose indices were sampled
  test_i = train_tree[val,] #The observations whose indices sampled
  
  for(i in 1:nset) {
    cat('Random Forest model: ',i,"; Number of Trees: ", ntreev[i],'\n')
    rffit = randomForest(train_i$Salesprice~., #Formula
                         data = train_i, #Data frame
                         ntree=ntreev[i], #Number of trees in the forest
                         maxnodes=15) #Maximum number of nodes in each tree
    #fmat[,i] = predict(rffit) #Predicted values for the fits
    
    #Calculating the MSE of current model
    aux = mean((test_i[,79]-predict(rffit, test_i))^2)
    
    #Store the current MSE
    out_MSE[j,i] = aux
  }
  
  #The union of the indices used currently and previously
  used = union(used,val)
  
  #The set of indices not used is updated
  set = (1:n)[-used]
  
  #Printing on the console the information that you want
  #Useful to keep track of the progress of your loop
  cat(j,"folds out of",kcv,'\n')
}

#Calculate the mean of MSE for each k
mMSE = apply(out_MSE, #Receive a matrix
             2, #Takes its columns (it would take its rows if this argument was 1)
             mean) #And for each column, calculate the mean

#Complexity x RMSE graph
plot(ntreev,sqrt(mMSE),
     xlab="# of trees",
     ylab="out-of-sample RMSE",
     col=4, #Color of line
     lwd=2, #Line width
     type="l", #Type of graph = line
     cex.lab=1.2, #Size of labs
     main=paste("kfold(",kcv,")")) #Title of the graph

best = which.min(mMSE)

#Including text at specific coordinates of the graph
text(ntreev[best],sqrt(mMSE[best])+0.0015, #Coordinates
     paste("# trees=",ntreev[best]), #The actual text
     col=2, #Color of the text
     cex=1.2) #Size of the text
text(10+100,sqrt(mMSE[1])-0.002,paste("# trees=",10),col=2,cex=1.2)
text(1000-100,sqrt(mMSE[4])+0.0015,paste("#trees=",1000),col=2,cex=1.2)

rffit = randomForest(train_tree$Salesprice~., #Formula
                     data = train_tree, #Data frame
                     ntree = 500, #Number of trees in the forest
                     maxnodes=15)

rf_predict = exp(predict(rffit,newdata=test_tree)) - 1

# construct data frame for solution
solution = data.frame(Id=as.integer(rownames(test_tree)),SalePrice=rf_predict)
write.csv(solution,"rf.csv",row.names=FALSE)

#Show important Random Forest variables
#imp_RF <- importance(rffit)
#imp_DF <- data.frame(Variables = row.names(imp_RF), MSE = imp_RF[,1])
#imp_DF <- imp_DF[order(imp_DF$MSE, decreasing = TRUE),]
#ggplot(imp_DF[1:20,], aes(x=reorder(Variables, MSE), y=MSE, fill=MSE)) + geom_bar(stat = 'identity')
#  + labs(x = 'Variables', y= '% increase MSE if variable is randomly permuted')
#  + coord_flip() + theme(legend.position="none")



##########Boosting


##########PLS
train$SalePrice <- y
n = dim(train)[1]

#Define the number of folds
kcv = 10 #In this case, 10-fold cross validation
#This k has nothing to do with the k from knn

#Size of the fold (which is the number of elements in the test matrix)
n0 = round(n/kcv, #Number of observations in the fold
           0) #Rounding with 0 decimals

set = 1:n
used = NULL
outMSE = NULL

for(j in 1:kcv){
  
  if(n0<length(set)){ #If the set of 'not used' is > than the size of the fold
    val = sample(set, size = n0) #then sample indices from the set
  }
  
  if(n0>=length(set)){ #If the set of 'not used' is <= than the size of the fold
    val=set #then use all of the remaining indices as the sample
  }
  
  #Create the train and test matrices
  train_t = train[-val,] #Every observation except the ones whose indices were sampled
  test_t = train[val,] #The observations whose indices sampled
  test_f <- subset(test_t, select=-c(SalePrice))
  
  #The current model
  pls.fit=plsr(SalePrice~., data=train_t,validation="CV")
  predplot(pls.fit, ncomp=3)
  pls.pred=predict(pls.fit,test_f,ncomp=3)
  
  #Calculating the MSE of current model
  aux = mean((pls.pred-test_t[,"SalePrice"])^2)
  outMSE = c(outMSE, aux)
  #print(aux)
  #The union of the indices used currently and previously
  used = union(used,val)
  
  #The set of indices not used is updated
  set = (1:n)[-used]
}
mMSE = mean(outMSE)
mMSE

pls.fit=plsr(SalePrice~., data=train,validation="CV")
pls.pred=exp(predict(pls.fit,test,ncomp=4))-1
title = c("SalePrice")
names(title)= c("Id")
write.table(title,file="pls.csv",col.names =F, sep = ",")
write.table(pls.pred,file="pls.csv",col.names =F, sep = ",", append = T)
