#################### LIBRARIES ####################
library('dplyr')
library('Rmisc')
library(pls)
library(caret)
library(stringr)
library(glmnet)
library(MASS)
library(mgcv)
library(nnet) 
library(rpart)
library(rpart.plot)
library(randomForest)


#################### VARIABLES ####################

# set the following seed:
#seed = 2928893

#################### ML FUNCTIONS ####################

### We will regularly need to shuffle a vector. This function
### does that for us.
shuffle = function(X){
  
  #set.seed(seed)
  new.order = sample.int(length(X))
  new.X = X[new.order]
  return(new.X)
}


# Get the folds for CV
get.folds = function(n, K) {
  
  ### Get the appropriate number of fold labels
  n.fold = ceiling(n / K) # Number of observations per fold (rounded up)
  fold.ids.raw = rep(1:K, times = n.fold) # Generate extra labels
  fold.ids = fold.ids.raw[1:n] # Keep only the correct number of labels
  
  ### Shuffle the fold labels
  #set.seed(seed)
  folds.rand = fold.ids[sample.int(n)]
  
  return(folds.rand)
}


### We will also often need to calculate MSE using an observed
### and a prediction vector. This is another useful function.
get.MSPE = function(Y, Y.hat){
  
  return(mean((Y - Y.hat)^2))
}


### Calculate RMSPEs
get.rmspe = function(all.MSPEs) {
  
  all.RMSPEs = apply(all.MSPEs, 1, function(W){
    
    best = min(W)
    return(W / best)
  })  
  
  return(t(all.RMSPEs))
  
}

# Rescale
rescale <- function(x1,x2){
  for(col in 1:ncol(x1)){
    a <- min(x2[,col])
    b <- max(x2[,col])
    x1[,col] <- (x1[,col]-a)/(b-a)
  }
  x1
}
  



### FUNCTIONS FOR CV ####

# Least Squares
fit.ls = function(data.train, data.valid, Y.valid, response_name) {
  
  formula = as.formula(paste(response_name, "~", "."))  
  
  fit.ls  = lm(formula, data=data.train)
  pred.ls = predict(fit.ls, newdata = data.valid)
  MSPE.ls = get.MSPE(Y.valid, pred.ls)
  
  return(MSPE.ls)
}


# Stepwise AIC
fit.step.aic = function(data.train, data.valid, Y.valid, response_name) {
  
  formula1 = as.formula(paste(response_name, "~", "1"))  
  formula2 = as.formula(paste(response_name, "~", "."))  
  
  fit.start = lm(formula1,   data = data.train)
  fit.end   = lm(formula2,   data = data.train)
  
  step.AIC = step(fit.start, list(upper = fit.end), k=2, trace = 0)
  pred.step.AIC = predict(step.AIC, data.valid)
  err.step.AIC = get.MSPE(Y.valid, pred.step.AIC)
  
  return(err.step.AIC)
}


# Stepwise BIC
fit.step.bic = function(data.train, data.valid, Y.valid, response_name) {
  
  formula1 = as.formula(paste(response_name, "~", "1"))  
  formula2 = as.formula(paste(response_name, "~", "."))  
  
  fit.start = lm(formula1,   data = data.train)
  fit.end   = lm(formula2,   data = data.train)
  
  step.BIC = step(fit.start, list(upper = fit.end), k = log(n.train), trace = 0)
  pred.step.BIC = predict(step.BIC, data.valid)
  err.step.BIC = get.MSPE(Y.valid, pred.step.BIC)
  
  return(err.step.BIC)
}


# Ridge Regression
fit.ridge.reg = function(data.train, data.valid, Y.valid, response_name) {
  
  formula = as.formula(paste(response_name, "~", "."))  
  
  lambda.vals = seq(from = 0, to = 100, by = 0.05)
  
  fit.ridge = lm.ridge(formula, lambda = lambda.vals, data = data.train)
  
  ind.min.GCV = which.min(fit.ridge$GCV)
  lambda.min = lambda.vals[ind.min.GCV]
  
  all.coefs.ridge = coef(fit.ridge)
  coef.min = all.coefs.ridge[ind.min.GCV,]
  
  matrix.valid.ridge = model.matrix(formula, data = data.valid)
  pred.ridge = matrix.valid.ridge %*% coef.min
  MSPE.ridge = get.MSPE(Y.valid, pred.ridge)
  return(MSPE.ridge)
  
}


# LASSO
fit.lasso.1se = function(data.train, data.valid, Y.valid, Y.train, response_name) {
  
  formula = as.formula(paste(response_name, "~", "."))  
  
  matrix.train.raw = model.matrix(formula, data = data.train)
  matrix.train = matrix.train.raw[, -1]
  
  all.LASSOs = cv.glmnet(x = matrix.train, y = Y.train)
  
  lambda.1se = all.LASSOs$lambda.1se
  
  coef.LASSO.1se = predict(all.LASSOs, s = lambda.1se, type='coef')
  
  included.LASSO.1se = predict(all.LASSOs, s = lambda.1se, type = "nonzero")
  
  matrix.valid.LASSO.raw = model.matrix(formula, data = data.valid)
  matrix.valid.LASSO = matrix.valid.LASSO.raw[,-1]
  
  pred.LASSO.1se = predict(all.LASSOs, newx = matrix.valid.LASSO, s = lambda.1se, type = "response")

  MSPE.LASSO.1se = get.MSPE(Y.valid, pred.LASSO.1se)
  
  return(MSPE.LASSO.1se)
  
  
}


### LASSO Min ###
fit.lasso.min = function(data.train, data.valid, Y.valid, Y.train, response_name) {
  
  formula = as.formula(paste(response_name, "~", "."))  
  
  matrix.train.raw = model.matrix(formula, data = data.train)
  matrix.train = matrix.train.raw[, -1]
  
  all.LASSOs = cv.glmnet(x = matrix.train, y = Y.train)
  
  lambda.min = all.LASSOs$lambda.min
  
  coef.LASSO.min = predict(all.LASSOs, s = lambda.min, type='coef')
  
  included.LASSO.min = predict(all.LASSOs, s = lambda.min, type = "nonzero")
  
  matrix.valid.LASSO.raw = model.matrix(formula, data = data.valid)
  matrix.valid.LASSO = matrix.valid.LASSO.raw[,-1]
  
  pred.LASSO.min = predict(all.LASSOs, newx = matrix.valid.LASSO, s = lambda.min, type = "response")
  
  MSPE.LASSO.min = get.MSPE(Y.valid, pred.LASSO.min)
  
  return(MSPE.LASSO.min)
  
}
  

# GAM
fit.gam.model = function(data.train, data.valid, Y.valid, response_name) {
  
  formula = as.formula(paste(response_name, "~", "s(X1)+s(X2)+s(X3)+
                                                  s(X4)+s(X5)+s(X6)+
                                                  s(X7)+s(X8)+s(X9)+
                                                  s(X10)+s(X11)+s(X12)+
                                                  s(X13)+s(X14)+s(X15)"))  
  
  fit.gam = gam(formula, data = data.train)  ## DYNAMIC
  
  pred.gam = predict(fit.gam, data.valid)
  MSPE.gam= get.MSPE(Y.valid, pred.gam) 
  
  return(MSPE.gam)
  
}


# PPR
fit.ppr.model = function(data.train, data.valid, K.ppr, max.terms, fold_index, response_name) {
  
  formula = as.formula(paste(response_name, "~", "."))  
  
  n.train = nrow(data.train)
  folds.ppr = get.folds(n.train, K.ppr)
  
  # Hold internal MSPEs
  MSPEs.ppr = array(0, dim = c(max.terms, K.ppr))
  
  ## CV ##
  for(j in 1:K.ppr){
    
    train.ppr = data.train[folds.ppr != j,]
    valid.ppr = data.train[folds.ppr == j,] 
    Y.valid.ppr = valid.ppr[, response_name]
    
    for(l in 1:max.terms){
      
      fit.ppr = ppr(formula, 
                    data = train.ppr, 
                    max.terms = max.terms, 
                    nterms = l, 
                    sm.method = "gcvspline")
      
      ### Get predictions and MSPE
      pred.ppr = predict(fit.ppr, valid.ppr)
      MSPE.ppr = get.MSPE(Y.valid.ppr, pred.ppr) 
      
      ### Store MSPE. Make sure the indices match for MSPEs.ppr
      MSPEs.ppr[l, j] = MSPE.ppr
    }
  }
  
  ### Get average MSPE for each number of terms
  ave.MSPE.ppr = apply(MSPEs.ppr, 1, mean)
  
  ### Get optimal number of terms
  best.terms = which.min(ave.MSPE.ppr)
  
  # print(paste0("PPR: Optimal Tuning Parameter on fold ", fold_index, ": ", best.terms))
  
  ### Fit PPR on the whole CV training set using the optimal number of terms 
  fit.ppr.best = ppr(formula, 
                     data = data.train,
                     max.terms = max.terms, 
                     nterms = best.terms, 
                     sm.method = "gcvspline")
  
  pred.ppr.best = predict(fit.ppr.best, data.valid)
  MSPE.ppr.best = get.MSPE(Y.valid, pred.ppr.best) 
  
  return(MSPE.ppr.best)
}


# 1 Layer Neural Net
fit.1layer.nnet = function(data.train.full, data.valid.full, M, K, all.n.hidden, all.shrink, response_name) {
  
  all.pars = expand.grid(n.hidden = all.n.hidden, shrink = all.shrink)
  n.pars  = nrow(all.pars) 
  CV.MSPEs = array(0, dim = c(K, n.pars))
  
  n = nrow(data.train.full)
  folds = get.folds(n, K)
  
  
  for(i in 1:K){
    
    ### Print progress update
    #print(paste0("-> Internal Nnet CV:", i, " of ", K))
    
    ### Split data and rescale predictors
    data.train  = data.train.full[folds != i,]
    X.train.raw = data.train[, !names(data.train) %in% c(response_name)]
    X.train = rescale(X.train.raw, X.train.raw)
    Y.train = data.train[, response_name]
    
    data.valid = data.train.full[folds == i,]
    X.valid.raw = data.valid[, !names(data.valid) %in% c(response_name)]
    X.valid = rescale(X.valid.raw, X.train.raw)
    Y.valid = data.valid[, response_name]
    
    
    ### Fit neural net models for each parameter combination. A second 
    for(j in 1:n.pars){
      
      ### Get current parameter values
      this.n.hidden = all.pars[j,1]
      this.shrink = all.pars[j,2]
      
      ### We need to run nnet multiple times to avoid bad local minima. Create
      ### containers to store the models and their errors.
      all.nnets = list(1:M)
      all.SSEs = rep(0, times = M)
      
      ### We need to fit each model multiple times. This calls for another
      ### for loop.
      for(l in 1:M){
        ### Fit model
        fit.nnet = nnet(X.train, Y.train, linout = TRUE, size = this.n.hidden,
                        decay = this.shrink, maxit = 1000, trace = FALSE)
        
        ### Get model SSE
        SSE.nnet = fit.nnet$value
        
        ### Store model and its SSE
        all.nnets[[l]] = fit.nnet
        all.SSEs[l] = SSE.nnet
      }
      
      ### Get best fit using current parameter values
      ind.best = which.min(all.SSEs)
      fit.nnet.best = all.nnets[[ind.best]]
      
      ### Get predictions and MSPE, then store MSPE
      pred.nnet = predict(fit.nnet.best, X.valid)
      MSPE.nnet = get.MSPE(Y.valid, pred.nnet)
      
      CV.MSPEs[i, j] = MSPE.nnet # Be careful with indices for CV.MSPEs
    }
  }
  
  # Get the most optimal model
  CV.MSPEs.mean    = colMeans(CV.MSPEs)
  mean.min         = which.min(CV.MSPEs.mean) 
  optimal.n.hidden = all.pars[mean.min, 1]
  optimal.shrink   = all.pars[mean.min, 2]
  
  X.train.raw.all =  data.train.full[, !names(data.train.full) %in% c(response_name)]
  X.train.all     = rescale(X.train.raw.all, X.train.raw.all)
  Y.train.all     = data.train.full[, response_name]
  
  X.valid.raw.all = data.valid.full[, !names(data.valid.full) %in% c(response_name)]
  X.valid.all     = rescale(X.valid.raw.all, X.train.raw.all)
  Y.valid.all     = data.valid.full[, response_name]  
  
  all.nnets = list(1:M)
  all.SSEs = rep(0, times = M)
  
  for(l in 1:M) {
    
    ### Fit model
    fit.nnet = nnet(X.train.all, Y.train.all, linout = TRUE, size = optimal.n.hidden,
                    decay = optimal.shrink, maxit = 1000, trace = FALSE)
    
    ### Get model SSE
    SSE.nnet = fit.nnet$value
    
    ### Store model and its SSE
    all.nnets[[l]] = fit.nnet
    all.SSEs[l] = SSE.nnet
    
  }  
  
  ### Get best fit using current parameter values
  ind.best = which.min(all.SSEs)
  fit.nnet.best = all.nnets[[ind.best]]
  
  ### Get predictions and MSPE, then store MSPE
  pred.nnet.valid = predict(fit.nnet.best, X.valid.raw.all)
  MSPE.nnet = get.MSPE(Y.valid.all, pred.nnet.valid)  
  
  print(paste0("-> Nnet - Most Optimal Hyper Parameters: ", optimal.n.hidden, ", ", 
               optimal.shrink, " - MSPE: ", 
               MSPE.nnet))
  
  return(MSPE.nnet)
  
}


# Regression Tree with CP = 0
reg.tree.cp.zero = function(data.train, data.valid, response_name) {
  
  formula  = as.formula(paste(response_name, "~", "."))  
  
  fit.tree = rpart(formula, data = data.train, cp = 0)
  
  pred.tree  = predict(fit.tree, data.valid)
  Y.valid    = data.valid[, response_name]
  MSPE.tree  = get.MSPE(Y.valid, pred.tree)
  
  return(MSPE.tree)
}


# Regression Tree with Min CV CP
reg.tree.cp.min = function(data.train, data.valid, response_name) {

  formula  = as.formula(paste(response_name, "~", "."))  
  
  fit.tree = rpart(formula, data = data.train, cp = 0)
  
  info.tree = fit.tree$cptable
  
  # CV ==> xerror
  ind.min = which.min(info.tree[,"xerror"])
  
  # Optimal CV level
  CP.min.raw = info.tree[ind.min, "CP"]
  
  # average out with row above
  if(ind.min == 1){
    
    ### If minimum CP is in row 1, store this value
    CP.min = CP.min.raw
    
  } else{
    
    CP.above = info.tree[ind.min-1, "CP"]
    CP.min = sqrt(CP.min.raw * CP.above)
    
  }
  
  # CP min tree
  fit.tree.cp.min = rpart(formula, data = data.train, cp = CP.min.raw)
  
  # predict and get MSPE
  pred.tree  = predict(fit.tree.cp.min, data.valid)
  Y.valid    = data.valid[, response_name]
  MSPE.tree  = get.MSPE(Y.valid, pred.tree)
  
  return(MSPE.tree)
  
}


# Regression Tree with Min CV CP
reg.tree.1se = function(data.train, data.valid, response_name) {
  
  formula   = as.formula(paste(response_name, "~", "."))  
  
  fit.tree  = rpart(formula, data = data.train, cp = 0)
  
  info.tree = fit.tree$cptable
  
  ind.min   = which.min(info.tree[,"xerror"])
  
  err.min   = info.tree[ind.min, "xerror"]
  se.min    = info.tree[ind.min, "xstd"]
  threshold = err.min + se.min
  
  ind.1se   = min(which(info.tree[1:ind.min, "xerror"] < threshold))
  
  CP.1se.raw = info.tree[ind.1se, "xerror"]
  
  if(ind.1se == 1){
    
    ### If best CP is in row 1, store this value
    CP.1se = CP.1se.raw
    
  } else{
    
    ### If best CP is not in row 1, average this with the value from the
    ### row above it.
    
    ### Value from row above
    CP.above = info.tree[ind.1se-1, "CP"]
    
    ### (Geometric) average
    CP.1se   = sqrt(CP.1se.raw * CP.above)
    
  }
  
  ### Prune the tree
  fit.tree.1se = prune(fit.tree, cp = CP.1se)
  
  # predict and get MSPE
  pred.tree  = predict(fit.tree.1se, data.valid)
  Y.valid    = data.valid[, response_name]
  MSPE.tree  = get.MSPE(Y.valid, pred.tree)
  
  return(MSPE.tree)
  
}


#################### VISUALIZATION FUNCTIONS ####################

plot.MSPE.boxplot = function(all.MSPEs) {
  
  par( mfrow = c(1,1) )
  boxplot(all.MSPEs, 
          main = paste0("CV MSPEs over ", K, " folds"), 
          las = 2)
  
  par(cex.axis=0.75)
  
}

plot.RMSPE.boxplot = function(all.RMSPEs) {
  
  boxplot(all.RMSPEs, 
          main = paste0("CV RMSPEs over ", K, " folds"), 
          las = 2,
          ylim = c(1,2))
  par(cex.axis=0.50)
  
}















