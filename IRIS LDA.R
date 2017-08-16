library(corrplot)
iris_train <- read.csv("train.csv")
iris_test <- read.csv("test.csv")

dim(iris_train)
dim(iris_test)
head(iris_train)
summary(iris_train)

#corrplot(iris_train[,-c(1,6)])

library(ggplot2)
install.packages("cowplot")
library(cowplot)

plot1<-qplot(Petal.Length, Petal.Width, data=iris_train, colour=Species)
plot2<-qplot(Sepal.Length, Sepal.Width, data=iris_train, colour=Species)
plot3<-qplot(Sepal.Length, Petal.Width, data=iris_train, colour=Species)
plot4<-qplot(Sepal.Length, Petal.Length, data=iris_train, colour=Species)
plot5<-qplot(Sepal.Width, Petal.Width, data=iris_train, colour=Species)
plot6<-qplot(Sepal.Width, Petal.Length, data=iris_train, colour=Species)

plot_grid(plot1,plot2,plot3,plot4,plot5,plot6, align='h')

# Versicolor and virginica are similar to each other

iris_train$New_Class <- 'VClass'

iris_train$New_Class[which(iris_train$Species == 'setosa')] <- 'setosa'

library(MASS)
fp1 <- lda(formula = New_Class ~ Sepal.Length+Sepal.Width+Petal.Length+Petal.Width, 
         data = iris_train) 
summary(fp1)
fp1

p_fp1 <- predict(object = fp1,
                newdata = iris_test)
summary(p_fp1)
p_fp1$x



iris_subset <- subset(iris_train, New_Class =='VClass')
fp2 <- lda(formula = Species ~ Sepal.Length+Sepal.Width+Petal.Length+Petal.Width, 
           data = iris_subset)

p_fp2 <- predict(object = fp2,newdata = iris_test)
p_fp2$x


fp_dataset = data.frame(Classify = iris_test[,"Species"],
                         f1 = p_fp1$x,f2 = p_fp2$x)


colnames(fp_dataset) <- c("Classify","FP1","FP2")


ggplot(fp_dataset) + geom_point(aes(FP1,FP2,colour = Classify, shape = Classify), size = 2.5)






