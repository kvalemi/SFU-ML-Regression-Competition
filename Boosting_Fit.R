#################################
library(gbm)

setwd('/Users/Kaveh/Documents/School Files/STAT 452/Assignments/Project 1')
source('Function Template.R')
Data = read.csv('Data2020.csv')

#################### LOAD DATA ####################

# Subset the data:
Data = Data[,c('Y','X12', 'X2', 'X15', 'X4', 'X14')]

### Set parameter values
### We will stick to resampling rate of 0.8, maximum of 10000 trees, and Tom's rule
### for choosing how many trees to keep.
max.trees = 10000
all.shrink = seq(from=0.045,to=0.046,by=0.0001)
all.depth = c(4)

all.pars = expand.grid(shrink = all.shrink, depth = all.depth)
n.pars = nrow(all.pars)

n = nrow(data)
K = 4

### Create folds
folds = get.folds(n, K)

### Create container for CV MSPEs
CV.MSPEs2 = array(0, dim = c(K, n.pars))

min.MSPE = 10

for(i in 1:K){
  ### Print progress update
  print(paste0(i, " of ", K))
  
  ### Split data
  data.train = Data[folds != i,]
  data.valid = Data[folds == i,]
  Y.valid = data.valid$Y
  
  
  ### Fit boosting models for each parameter combination
  for(j in 1:n.pars){
    
    ### Get current parameter values
    this.shrink = all.pars[j,"shrink"]
    this.depth = all.pars[j,"depth"]
    
    ### Fit model using current parameter values.
    fit.gbm = gbm(Y ~ ., 
                  data = data.train, 
                  distribution = "gaussian", 
                  n.trees = max.trees, 
                  interaction.depth = this.depth, 
                  shrinkage = this.shrink, 
                  bag.fraction = 0.8)
    
    ### Choose how many trees to keep using Tom's rule. This will print many
    ### warnings about not just using the number of trees recommended by
    ### gbm.perf(). We have already addressed this problem though, so we can
    ### just ignore the warnings.
    n.trees = gbm.perf(fit.gbm, plot.it = F) * 2
    
    ### Check to make sure that Tom's rule doesn't tell us to use more than 1000
    ### trees. If it does, add extra trees as necessary
    if(n.trees > max.trees){
      extra.trees = n.trees - max.trees
      fit.gbm = gbm.more(fit.gbm, extra.trees)
    }
    
    ### Get predictions and MSPE, then store MSPE
    pred.gbm = predict(fit.gbm, data.valid, n.trees)
    MSPE.gbm = get.MSPE(Y.valid, pred.gbm)
    
    CV.MSPEs2[i, j] = MSPE.gbm # Be careful with indices for CV.MSPEs
    
    if(MSPE.gbm < min.MSPE) {
      
      min.MSPE = MSPE.gbm
      print(paste0("Minimum MSPE - ", MSPE.gbm, " - acheived by parameters: (shrinkage, depth) --> ", this.shrink, ', ', this.depth))
      
    }
    
  }
}

names.pars = paste0(all.pars$shrink,"-", all.pars$depth)
colnames(CV.MSPEs2) = names.pars


