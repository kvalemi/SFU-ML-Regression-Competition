setwd('/Users/Kaveh/Documents/School Files/STAT 452/Assignments/Project 1')
data = read.csv('Data2020.csv')

# check the correlation coefficients
(cor(data, data$Y))

# X12, X10 have the strongest correlations

# Maybe we should transform or remove:
  # X14, X13, X11, X9, X8, X5, X3

# Possible transformations
# data$X14 = log(data$X14)
# data$X13 = log(data$X14)

# we could also scale all of the explanatory variables



# I am going to fit the model to three versions of the data:
  # 1) Raw Data
  # 2) Removing the variable mentioned in the RF


### PCA Analysis
data.matrix.raw = model.matrix(Y ~ ., data = Data)
data.matrix = data.matrix.raw[,-1]

fit.PCA = prcomp(data.matrix, scale. = T)
print(fit.PCA)

vars = fit.PCA$sdev^2

plot(1:length(vars), vars, main = "Scree plot", 
     xlab = "Principal Component", ylab = "Variance Explained")
abline(h = 1)

# Use 7 PC
all.PCs = fit.PCA$rotation
Data_dr_pca = all.PCs[,1:7]





