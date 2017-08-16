library(e1071)
mushroom_train <-read.csv("train.csv")
mushroom_test <-read.csv("test.csv")

mushroom_train<-mushroom_train[,-1]
mushroom_test<-mushroom_test[,-1]
head(mushroom_train)
head(mushroom_test)

summary(mushroom_train)

lambda <- c(0:50)

train_acc <- rep(NA,51)
test_acc <-rep(NA,51)

for (i in 1:length(lambda))
{
mushroom_nbc <- naiveBayes(V1~.,data =mushroom_train, laplace =lambda[i])
predict_train <- predict(mushroom_nbc, mushroom_train[,-1])
predict_test <- predict(mushroom_nbc, mushroom_test[,-1])
conf_mat_train <-table(predict_train,mushroom_train$V1)
conf_mat_test <-table(predict_test,mushroom_test$V1)
train_acc[i]<- (conf_mat_train[1,1]+conf_mat_train[2,2])/length(predict_train)
test_acc[i]<- (conf_mat_test[1,1]+conf_mat_test[2,2])/length(predict_test)
}

lambda_max_test_acc <- which(test_acc==max(test_acc)) -1

plot(lambda,train_acc,type="l",lty=1,lwd=3,xlab="Laplacian Smoothing Constant",ylab='Accuracy',main='Naive Bayes Accuracy Plot',col='blue',ylim = c(0.91,0.96))
lines(lambda,test_acc,"l",lty=1,lwd=3,col='violet')
legend('right',c('Training Set','Test Set'),pch=17, col=c('blue','violet'),text.col=c('blue','violet'),cex =0.6)



library(tree)

dim(mushroom_train)

mc<-seq(4,64,4)
train_acc_dt <- rep(NA,16)
test_acc_dt <-rep(NA,16)
#dtree1
#summary(dtree1)

#plot(dtree1)
#text(dtree1)
for (j in 1:length(mc))
{
dtree1 <- tree(V1~.,data=mushroom_train,control=tree.control(nobs=nrow(mushroom_train),minsize =mc[j]))

predict_train_dt <- predict(dtree1,mushroom_train,type="class")
#head(predict_train_dt)

predict_test_dt <-predict(dtree1,mushroom_test,type="class")
conf_mat_train_dt <-table(predict_train_dt,mushroom_train$V1)
conf_mat_test_dt<-table(predict_test_dt,mushroom_test$V1)
train_acc_dt[j]<- (conf_mat_train_dt[1,1]+conf_mat_train_dt[2,2])/length(predict_train_dt)
test_acc_dt[j]<- (conf_mat_test_dt[1,1]+conf_mat_test_dt[2,2])/length(predict_test_dt)
}

plot(mc,train_acc_dt,type="l",lty=1,lwd=3,xlab="Size Threshold",ylab='Accuracy',main='Decision Tree Accuracy Plot',col='blue',ylim = c(0.91,0.998))
lines(mc,test_acc_dt,"l",lty=1,lwd=3,col='violet')
legend('right',c('Training Set','Test Set'),pch=17, col=c('blue','violet'),text.col=c('blue','violet'),cex =0.6)
