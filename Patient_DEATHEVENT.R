library(ISLR)
library(readr)
dataset = read.csv("C:/Users/KimJiBeom/Desktop/데사 프로젝트/sample.csv")

library(MASS)


# using holdout (66.66% of the data) cross validation
sample_dat <- sample(1:nrow(dataset),round(nrow(dataset)/3,0), replace = FALSE) #train_set 0.3333, test_set 0.6666
test_dat <- setdiff(1:nrow(dataset), sample_dat)

train_set <- dataset[sample_dat,]
test_set <- dataset[test_dat,]

#LR
glm.fit = glm(DEATH_EVENT~age+anaemia+creatinine_phosphokinase+diabetes+ejection_fraction+high_blood_pressure+platelets+serum_creatinine+serum_sodium+sex+smoking+time, data=train_set)
summary(glm.fit)

# Prediction on training data
LR_train_pred = predict(glm.fit,type="response")

# Evaluate the performance of the logistic regression model on training set
LR_train_result<-ifelse(LR_train_pred>0.5,1,0)
str(LR_train_result)
mean(LR_train_result==train_set)

# Prediction on test data
LR_test_pred = predict(glm.fit, newdata=test_set,type="response")
LR_test_result <- ifelse(LR_test_pred>0.5,1,0)
str(LR_test_result)
mean(LR_test_result==test_set)

#ROC
library(ROCR)
p <- predict(glm.fit, newdata=test_set, type="response")
pr <- prediction(p, test_set$DEATH_EVENT)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc

# LDA
library(MASS)

lda_fit=lda(formula=DEATH_EVENT~., data=train_set)
lda_fit

LDA_train_pred = predict(lda_fit, train_set)
LDA_train_class = LDA_train_pred$class

table(train_set$DEATH_EVENT,LDA_train_class)
mean(train_set$DEATH_EVENT==LDA_train_class)

LDA_test_pred = predict(lda_fit, test_set)
LDA_test_class = LDA_test_pred$class

table(test_set$DEATH_EVENT,LDA_test_class)
mean(test_set$DEATH_EVENT==LDA_test_class)

#QDA
qda_fit = qda(formula=DEATH_EVENT~., data=train_set)
qda_fit

QDA_train_pred = predict(qda_fit, train_set)
QDA_train_class = QDA_train_pred$class

table(train_set$DEATH_EVENT,QDA_train_class)
mean(train_set$DEATH_EVENT==QDA_train_class)

QDA_test_pred = predict(qda_fit, test_set)
QDA_test_class = QDA_test_pred$class

table(test_set$DEATH_EVENT,QDA_test_class)
mean(test_set$DEATH_EVENT==QDA_test_class)

#KNN
library(class)
library(readr)

x = train_set[,-1]
y = train_set$DEATH_EVENT
x_test = test_set[,-1]
y_test = test_set$DEATH_EVENT
dat = data.frame(x=x, y=y)
dat_test = data.frame(x=x_test, y=y_test)



set.seed(300)
knn1 <- knn(train = x,test =x_test,y,k=1)
table(knn1,dat_test$y)
mean(knn1==dat_test$y)

knn3 <- knn(train = x,test =x_test,y,k=3)
table(knn3,dat_test$y)
mean(knn3==dat_test$y)

knn5 <- knn(train = x,test =x_test,y,k=5)
table(knn5,dat_test$y)
mean(knn5==dat_test$y)

knn7 <- knn(train = x,test =x_test,y,k=7)
table(knn7,dat_test$y)
mean(knn7==dat_test$y)

knn9 <- knn(train = x,test =x_test,y,k=9)
table(knn9,dat_test$y)
mean(knn9==dat_test$y)

#SVM
library(ISLR)
library(MASS)
library(readr)
library(e1071)

par(mfrow=c(1,1))
svmfit=svm(y~.,data=dat,kernel="radial",cost=10,gamma=1)
table(svmfit$fitted, dat$y)
mean(svmfit$fitted==dat$y)

dat_test = data.frame(x=x_test, y=as.factor(y_test))
pred.svm = predict(svmfit, newdata = dat_test)
table(pred.svm, dat_test$y)
mean(pred.svm==dat_test$y)

#Random Forest
library(caret)
library(randomForest)
library(readr)

rf<-randomForest(y~.,data=dat)

rf

table(actual=y_test,predicted=predict(rf,dat_test,type="class"))

mean(predict(rf,dat_test,type="class")==y_test)

#Neural Networks
library(nnet)

nn<-nnet(y~.,data=dat,size=10,decay=0.1)

table(actual=y,predicted=predict(nn,type="class"))

mean(y==predict(nn,type="class"))

nn<-nnet(y~.,data=dat_test,size=10,decay=0.1)

table(actual=y_test,predicted=predict(nn,newdata=dat_test,type="class"))

mean(y_test==predict(nn,newdata=dat_test,type="class"))

#Bagging
library(adabag)

bag<-bagging(y~.,data=dat)
summary(bag)
table(actual=y,predicted=predict(bag,newdata=dat)$class)
mean(y==predict(bag,newdata=dat)$class)
table(actual=y_test,predicted=predict(bag,newdata=dat_test)$class)
mean(y_test==predict(bag,newdata=dat_test)$class)

