#library
library(mlbench)
library(leaps)
library(rpart)
library(rpart.plot)
library(stats)
library(gplots)
library(ggplot2)
library(boot)
library(caret)
library(lattice)
library(adabag)
library(randomForest)
library(ROCR)

#Model eveluation

findAcracy <- function(x){
  accuracy = 0
  x.matrix <- as.matrix(x)
  for(i in 1:nrow(x.matrix)){
    for(j in 1:ncol(x.matrix)){
      if(i == j){
        accuracy = accuracy+x.matrix[i,j]
      }
    }
  }
  return(accuracy/sum(as.matrix(x)))
}


findRecall <- function(x){
  x.matrix <- as.matrix(x)
  return(x.matrix[1,1]/sum(x.matrix[1,1]+x.matrix[2,1]))
}

findFP <- function(x){
  x.matrix <- as.matrix(x)
  return(x.matrix[1,2]/sum(x.matrix[1,2]+x.matrix[2,2]))
}

findSpcfity <- function(x){
  x.matrix <- as.matrix(x)
  return(x.matrix[2,2]/sum(x.matrix[1,2]+x.matrix[2,2]))
}

findPcision <- function(x){
  x.matrix <- as.matrix(x)
  return(x.matrix[1,1]/sum(x.matrix[1,1]+x.matrix[1,2]))
}

Fscore <- function(x){
  x.matrix <- as.matrix(x)
  precision<-findPcision(x)
  recall<- findRecall(x)
  return((2*precision*recall)/(precision+recall))
}

bank.full <-  read.csv(file.choose(),header = TRUE,sep=";")
bank.partial  <-  read.csv(file.choose(),header = TRUE,sep = ";")
ncol(bank.partial)
#Finding missing values
sum(is.na(bank.partial))
# There is no missing values

#Change to numeric
attach(bank.partial)
bank.partial$job <- as.numeric(bank.partial$job)
bank.partial$marital <- as.numeric(bank.partial$marital)
bank.partial$education <- as.numeric(bank.partial$education)
bank.partial$default <- as.numeric(bank.partial$default)
bank.partial$housing <- as.numeric(bank.partial$housing)
bank.partial$loan <- as.numeric(bank.partial$loan)
bank.partial$contact <- as.numeric(bank.partial$contact)
bank.partial$month <- as.numeric(bank.partial$month)
bank.partial$day_of_week <- as.numeric(bank.partial$day_of_week)
bank.partial$poutcome <- as.numeric(bank.partial$poutcome)
bank.partial$y <- as.numeric(bank.partial$y)

#Change 1 to 0 and 2 to 1 since logisitic regression requires range between 0 to 1
bank.partial$y[bank.partial$y==1] <- 0
bank.partial$y[bank.partial$y==2] <- 1



#linear regression
lm.fit = lm(y~. , data=bank.partial)
summary(lm.fit)
plot(lm.fit)

#forward selection
bank.lm.null <- glm(y~1, data = bank.partial)
bank.lm.full <- glm(y~., data = bank.partial)
step(bank.lm.null, scope=list(lower=bank.lm.null, upper=bank.lm.full), direction="forward")

lm.fit.2 = lm(y ~ duration + nr.employed + pdays + month + cons.conf.idx + poutcome + previous + contact + cons.price.idx + emp.var.rate + education, data=bank.partial)
summary(lm.fit.2)
plot(lm.fit.2)
lm <- lm(y~loan, data = bank.partial)
summary(lm)


#Split data
ind = sample(2, nrow(bank.partial), replace=TRUE, prob=c(0.7, 0.3))
bank.train = bank.partial[ind==1,]
nrow(bank.train)
bank.test = bank.partial[ind==2,]
nrow(bank.test)

###LDA
library(klaR)
#c("duration", "nr.employed", "pdays" ,"month", "cons.conf.idx", "contact", "poutcome" , "campaign")
partimat(bank.partial[,c("duration","pdays")], unclass(bank.partial[,"y"]), method="lda")
# easy way to draw LDA or QDA


LDA.fit = lda(y ~ ., data = bank.train)
LDA.fit
plot(LDA.fit)
plot(LDA.fit, col = as.integer(bank.train$y))


LDA.pred = predict(LDA.fit, bank.train)
table(bank.train$y, LDA.pred$class)

# Test Data
LDA.pred1 = predict(LDA.fit, bank.test)
names(LDA.pred1)

head(LDA.pred1$class,10)				# Maximum a Posteriori Probability Classification
head(round(LDA.pred1$posterior,3),10)	# Posterior Probabilities for the Classes

table(bank.test$y, LDA.pred1$class)	## Confusion Matrix & Overall fraction of correct predictions
mean(bank.test$y == LDA.pred1$class) 

# 1st Discriminant function
plot(LDA.pred$x[,1], col=c("blue","green","red")[bank.train$y], pch=19)
# 2nd Discriminant function
plot(LDA.pred$x[,2], col=c("blue","green","red")[bank.train$y], pch=19)
# X= 1st, Y = 2nd Discriminant function?
plot(LDA.pred$x[,1],LDA.pred$x[,2], col=c("blue","green","red")[bank.train$y], pch=19)


#Logistic regression

#forward stepwise selection
bank.glm.null <- glm(y~1, data = bank.train, family = binomial)
bank.glm.full <- glm(y~., data = bank.train, family = binomial)
step(bank.glm.null, scope=list(lower=bank.glm.null, upper=bank.glm.full), direction="forward")

#y ~ duration + nr.employed + pdays + month + cons.conf.idx + contact + poutcome + campaign
set.seed(123)
bank.glm <- glm(y ~ duration + nr.employed + pdays + month + cons.conf.idx + contact + poutcome + campaign, data = bank.train, family = binomial)
summary(bank.glm)
bank.glm.probs = predict(bank.glm,type="response")
bank.glm.probs[1:10]
bank.glm.pred=rep(0,nrow(bank.train))
bank.glm.pred[bank.glm.probs >.5]=1
t.bank.train = table(bank.glm.pred,bank.train$y)
t.bank.train
findRecall(t.bank.train)
#Specificity Rate
findSpcfity(t.bank.train)

#to find the error rate for the model 
library(boot)
cv.error.1 <- cv.glm(bank.train,bank.glm,K=10)
cv.error.1$delta

#try it with test data
bank.glm.test.probs=predict(bank.glm , newdata = bank.test,type="response")
bank.glm.test.pred=rep(0,nrow(bank.test))
bank.glm.test.pred[bank.glm.test.probs >.5]=1
t.bank.test = table(bank.glm.test.pred, bank.test$y)

t.bank.test

#increasing row of variable y equal to 0

data.class.0 <-NULL
data.class.1 <-NULL
data.class.0 <- subset(bank.partial, bank.partial$y == 0)
data.class.1 <- subset(bank.partial, bank.partial$y == 1)
nrow(data.class.0)
##3668
nrow(data.class.1)
##451
data.bank <- NULL
data.bank <- data.class.0

for(i in 1:7){
data.bank <- rbind(data.bank, data.class.1)
}
nrow(data.bank)
table(data.bank$y)

#split data one more time

ind1 = sample(2, nrow(data.bank), replace=TRUE, prob=c(0.7, 0.3))
data.train = data.bank[ind1==1,]
nrow(data.train)
data.test = data.bank[ind1==2,]
nrow(data.test)

# Foward stepwise selection

data.glm.null <- glm(y~1, data = data.train, family = binomial)
data.glm.full <- glm(y~., data = data.test, family = binomial)
step(data.glm.null, scope=list(lower=data.glm.null, upper=data.glm.full), direction="forward")

#y ~ duration + nr.employed + pdays + month + cons.conf.idx + contact + poutcome + campaign

data.glm <- glm(y ~ duration + nr.employed + pdays + month + cons.conf.idx + 
                  emp.var.rate + euribor3m + education + contact + cons.price.idx + 
                  job + poutcome + marital + age + campaign + day_of_week, data = data.train, family = binomial)
summary(data.glm)
data.glm.probs = predict(data.glm,type="response")
data.glm.probs[1:10]
data.glm.pred=rep(0,nrow(data.train))
data.glm.pred[data.glm.probs >.5]=1
t.data.train = table(data.glm.pred,data.train$y)

findAcracy(t.data.train)
findRecall(t.data.train)
findSpcfity(t.data.train)

library(boot)

cv.error.data <- cv.glm(data.train,data.glm,K=10)
cv.error.data$delta

#try it with test data
data.glm.test.probs=predict(data.glm , newdata = bank.test,type="response")
data.glm.test.pred=rep(0,nrow(data.test))
data.glm.test.pred[data.glm.test.probs >.5]=1
t.data.test = table(data.glm.test.pred, data.test$y)

t.data.test
findRecall(t.data.test)
findSpcfity(t.data.test)

### Decision trees
library(rpart)
library(rpart.plot)

rpart.fit = rpart(y ~., data = data.train,method = "class",control = rpart.control(cp = 0 , minbucket = 20))
rpart.plot(rpart.fit)
t.rpart = table(predict(rpart.fit,type = "class"), data.train$y, dnn = c("Predicted", "Actual")) 
t.rpart
t.test.rpart = table(predict(rpart.fit, newdata = data.test,type = "class"), data.test$y, dnn = c("Predicted", "Actual")) 
t.test.rpart
findSpcfity(t.test.rpart)
opt <- which.min(rpart.fit$cptable[,"xerror"])
cp <- rpart.fit$cptable[opt, "CP"] #lowest cp is 0

#Prunning

rpart.fit.prune <- prune(rpart.fit,cp = cp)
rpart.plot(rpart.fit.prune)
t.rpart.prune = table(predict(rpart.fit.prune,type = "class"), data.train$y, dnn = c("Predicted", "Actual")) 
t.rpart.prune
findSpcfity(t.rpart.prune)
t.test.rpart.prune = table(predict(rpart.fit.prune, newdata = data.test,type = "class"), data.test$y, dnn = c("Predicted", "Actual")) 
t.test.rpart.prune
findSpcfity(t.test.rpart.prune)

###Bagging
bagging.fit = randomForest(as.factor(y)~., data = data.train, ntree = 1000, proximity = TRUE, replace= TRUE, importance = TRUE, mtry = ncol(data.bank))
pred.bagging.train <- predict(bagging.fit,data.train,type = "class")
t.bagging = table(pred.bagging.train, data.train$y, dnn = c("Predicted", "Actual"))
t.bagging
#Recall Rate
findRecall(t.bagging)
#Specificity Rate
findSpcfity(t.bagging)

pred.bagging.test <- predict(bagging.fit , newdata = data.test, type = "class")
t.bagging.test = table(pred.bagging.test, data.test$y, dnn = c("Predicted", "Actual"))
t.bagging.test
#Recall Rate
findRecall(t.bagging.test)
#Specificity Rate
findSpcfity(t.bagging.test) ###seem overfitting 

###Random forest
rf.fit = randomForest(as.factor(y)~., data = data.train, ntree = 500, proximity = TRUE, replace= TRUE, importance = TRUE, mtry = sqrt(ncol(data.bank)))
pred.rf.train <- predict(rf.fit,data.train,type = "class")
t.rf = table(pred.rf.train, data.train$y, dnn = c("Predicted", "Actual"))
t.rf
#Recall Rate
findRecall(t.rf)
#Specificity Rate
findSpcfity(t.rf)

pred.rf.test <- predict(rf.fit , newdata = data.test, type = "class")
t.rf.test = table(pred.rf.test, data.test$y, dnn = c("Predicted", "Actual"))
t.rf.test
#Recall Rate
findRecall(t.rf.test)
#Specificity Rate
findSpcfity(t.rf.test) 


#Use anova for hypothesis
h1.aov <- aov(y~education, data=data.bank)
summary(h1.aov)
plotmeans(data.bank$education~data.bank$y)

h2.aov <- aov(y~loan, data=data.bank)
summary(h2.aov)
plotmeans(data.bank$loan~data.bank$y)


rf.fit.1 = randomForest(as.factor(y)~., data = bank.full, ntree = 500, proximity = TRUE, replace= TRUE, importance = TRUE, mtry = sqrt(ncol(data.bank)))

# adaboost

data.train$y <- as.factor(data.train$y)
adaboost.fit <- boosting(y~., data = data.train[,-11], mfinal = 30, control = rpart.control(maxdepth = 15, cp=-1))
barplot(adaboost.fit$importance[order(adaboost.fit.over$importance, decreasing  = TRUE)], ylim = c(0,100), main = "Variables Relative Importance", col = "lightblue")
table(adaboost.fit$class,data.train$y,dnn=c("Predicted Class","Observed Class"))
manipulator.predboosting <- predict.boosting(adaboost.fit,newdata = data.test)
pred = prediction(manipulator.predboosting$prob[,2], data.test$y) 
perf = performance(pred,"tpr","fpr")
plot(perf,xlab="False Positive Rate (False Alarm)",ylab="True Positive Rate (Recall or Sensitivity)")
auc = performance(pred, "auc") 
auc = unlist(slot(auc, "y.values"))
aucVal = paste(c("AUC = "), auc, sep = "") 
legend(0.5, 0.5, aucVal, cex = 0.7, box.col = "white")
abline(a= 0, b=1) 



##SVM
set.seed(1)
library(e1071)
set.seed(1)
tune.out=tune(svm ,y ~.,data=data.train ,kernel ="radial", ranges=list(cost=c(10,100) ,gamma=c(0.5,1,2,4)))
svmfit=svm(y ~., data=data.train,type="C-classification",cost=10, gamma=2)
summary(svmfit)
table(svmfit$fitted , data.train$y)
prd = predict(svmfit,newdata = data.test)
t.svm = table(prd,data.test$y)

findAcracy(t.svm)
findSpcfity(t.svm)
findRecall(findRecall)






bank.full$job <- as.numeric(bank.full$job)
bank.full$marital <- as.numeric(bank.full$marital)
bank.full$education <- as.numeric(bank.full$education)
bank.full$default <- as.numeric(bank.full$default)
bank.full$housing <- as.numeric(bank.full$housing)
bank.full$loan <- as.numeric(bank.full$loan)
bank.full$contact <- as.numeric(bank.full$contact)
bank.full$month <- as.numeric(bank.full$month)
bank.full$day_of_week <- as.numeric(bank.full$day_of_week)
bank.full$poutcome <- as.numeric(bank.full$poutcome)
bank.full$y <- as.numeric(bank.full$y)

#Change 1 to 0 and 2 to 1 since logisitic regression requires range between 0 to 1
bank.full$y[bank.full$y==1] <- 0
bank.full$y[bank.full$y==2] <- 1

#increasing row of variable y equal to 0

full.class.0 <-NULL
full.class.1 <-NULL
full.class.0 <- subset(bank.full, bank.full$y == 0)
full.class.1 <- subset(bank.full, bank.full$y == 1)
nrow(full.class.0)
##36548
nrow(full.class.1)
##4640
data.full1 <- NULL
data.full1 <- full.class.0

for(i in 1:7){
  data.full1 <- rbind(data.full1, full.class.1)
}
nrow(data.full1)
table(data.full1$y)

#split data one more time

ind2 = sample(2, nrow(data.full1), replace=TRUE, prob=c(0.7, 0.3))
full.train = data.full1[ind2==1,]
nrow(full.train)
full.test = data.full1[ind1==2,]
nrow(full.test)

svmfit1=svm(y ~., data=full.train,type="C-classification",cost=10, gamma=2)
summary(svmfit1)

table(svmfit1$fitted , full.train$y)
prd1 = predict(svmfit1,newdata = full.test)
t.svm1 = table(prd1,full.test$y)







