#################### Libraries ####################
library(randomForest)

#################### Load data ####################

# Set the directory (change directory to local directory of files on your own computer)
setwd('/Users/Kaveh/Documents/School Files/STAT 452/Assignments/Project 1')

# Read in the training data
training_Data = read.csv('Data2020.csv')

# Subset the training_Data to include only important features
training_Data = training_Data[,c('Y','X12', 'X15', 'X4')]

#################### Model Building ####################

# Define the optimal hyperparameters for the random forest
tuned.mtry       = 3 
tuned.nodesize   = 8 
tuned.samplesize = 0.50 

# Set the seed right before the random forest model
set.seed = 12
# Build the Random Forest model
fit.rf = randomForest(Y ~ ., 
                      data = training_Data, 
                      importance = F,
                      mtry = tuned.mtry, 
                      nodesize = tuned.nodesize,
                      ntree = 1000,
                      sampsize = tuned.samplesize * nrow(training_Data))

# Read in the test data
test_Data = read.csv('Data2020testX.csv')

# Subset test data to only include important features
test_Data = test_Data[,c('X12', 'X15', 'X4')]

# Predict Y from the test data
test_output = predict(fit.rf, test_Data)

# Output the data
write.table(test_output, file = './Data_Test_Output.csv', row.names = FALSE, col.names = FALSE)



    






