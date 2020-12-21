#################### LOAD LIBRARIES AND FUNCTIONS ####################
setwd('/Users/Kaveh/Documents/School Files/STAT 452/Assignments/Project 1')
source('Function Template.R')

#################### LOAD DATA ####################

Data = read.csv('Data2020.csv')
response_name = "Y"  ## DYNAMIC

#################### DATA MANIPULATION ####################
setwd('/Users/Kaveh/Documents/School Files/STAT 452/Assignments/Project 1')
source('Function Template.R')
Data = read.csv('Data2020.csv')

#################### LOAD DATA ####################

# Subset the data:
Data = Data[,c('Y','X12', 'X15', 'X4')]

#################### CROSS VALIDATION ####################

iters = 5
K = 10
n = nrow(Data)

all.models = c("LS", "stepwise.AIC", "stepwise.BIC", "Ridge", "LASSO-Min", "LASSO-1se", "PPR", "tree.cp.zero", "tree.cp.min", "tree.1SE", "NN-1layer")
all.MSPEs = array(0, dim = c(K*iters, length(all.models)))
colnames(all.MSPEs) = all.models

index = 0

for(j in 1:iters) {
  
  folds = get.folds(n, K)
  
  # lets actually run the CV
  for(i in 1:K) {
    
    print(paste0("-- Running Fold: ", i, " --"))
    
    ### Set up the training data for folds ###
    data.train = Data[folds != i,]
    data.valid = Data[folds == i,]
    Y.train    = data.train[, response_name]
    Y.valid    = data.valid[, response_name]
    n.train    = nrow(data.train)
    
    
    ### Least Squares ###
    all.MSPEs[index, "LS"] = fit.ls(data.train, data.valid, Y.valid, response_name)
    
    #### Step ###
    #all.MSPEs[index, "stepwise.AIC"] = fit.step.aic(data.train, data.valid, Y.valid, response_name)
    #all.MSPEs[index, "stepwise.BIC"] = fit.step.bic(data.train, data.valid, Y.valid, response_name)
    
    ### Ridge Reg ###
    #all.MSPEs[index, "Ridge"] = fit.ridge.reg(data.train, data.valid, Y.valid, response_name)
    
    ### LASSO ###
    #all.MSPEs[index, "LASSO-Min"] = fit.lasso.min(data.train, data.valid, Y.valid, Y.train, response_name)
    #all.MSPEs[index, "LASSO-1se"] = fit.lasso.1se(data.train, data.valid, Y.valid, Y.train, response_name)  
    
    ### Fitting a GAM ###
    #all.MSPEs[index, "GAM"] = fit.gam.model(data.train, data.valid, Y.valid, response_name)
    
    ### Fitting a PPR ###
    #all.MSPEs[index, "PPR"] = fit.ppr.model(data.train, data.valid, 5, 5, i, response_name)  ## DYNAMIC  
    
    ### NN ###
    #all.n.hidden = c(5,6)
    #all.shrink   = c(1.3, 1.32, 1.34, 1.36, 1.38, 1.4)
    #all.MSPEs[index, "NN-1layer"] = fit.1layer.nnet(data.train, data.valid, 5, 5, all.n.hidden, all.shrink, response_name)
    
    # Tree 
    #all.MSPEs[index, "tree.cp.zero"] = reg.tree.cp.zero(data.train, data.valid, response_name)
    #all.MSPEs[index, "tree.cp.min"]  = reg.tree.cp.min(data.train, data.valid, response_name)
    #all.MSPEs[index, "tree.1SE"]     = reg.tree.1se(data.train, data.valid, response_name)
    
    # Random Forests
    
  
    index = index + 1
  }
  
  # print(all.MSPEs)

}

# print the MSPES + boxplots
plot.MSPE.boxplot(all.MSPEs)

### Calculate RMSPEs
all.RMSPEs = get.rmspe(all.MSPEs)
plot.RMSPE.boxplot(all.RMSPEs)








