getwd()
#extract the file names fro directory
file_names <- dir(getwd()) 

#Separating train and test files names
train_files<-file_names[which(grepl("train",file_names) == TRUE)]
test_files<-file_names[which(grepl("test",file_names) == TRUE)]

# Creating training and test set by reading files

label_train<- vector("numeric")
label_test <- vector("numeric")
training_data <- data.frame()
testing_data <- data.frame()

for(i in (1:length(train_files)))
{ data_train<-read.csv(train_files[i],header=TRUE)
  data_test<-read.csv(test_files[i],header=TRUE)
  training_data <- rbind(training_data,data_train)
  testing_data <- rbind(testing_data,data_test)
  label_train <- c(label_train,rep(i-1,nrow(data_train)))
  label_test <- c(label_test,rep(i-1,nrow(data_test)))
}

table(label_train)
table(label_test)
# Merging data and label vectors to create complete training and test set
training_data <- cbind(training_data,label_train)
testing_data <- cbind(testing_data,label_test)
#Removing 1st id column
training_data<-training_data[,-1]
testing_data<-testing_data[,-1]


#Funtion to omit NA value rows , add small value to columns with all zero and scale the dataframe

clean <- function(df){
 cat("Input NA values:",sum(is.na(df)),"\n")
 if (sum(is.na(df))>0){
   df <- na.omit(df)
   cat("Resultant NA values:",sum(is.na(df)))
 }
  df[,-ncol(df)] <- df[,-ncol(df)] + rnorm(nrow(df)*(ncol(df)-1),0.001,0.001)
  df <- df[,colSums(df != 0)>0] 
  df[,-ncol(df)] = scale(df[,-ncol(df)])
  df<-as.data.frame(df)
  
}
      
# Calling clean function on training and test data
training_data<- clean(training_data)
testing_data<-clean(testing_data)
