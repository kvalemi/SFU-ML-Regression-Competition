##### Log of Performance ########

# The full raw data is bad so we should subset it

# Order of subsets with lowest average OOB MSPE:
  # Data = Data[,c('Y','X12', 'X15', 'X4')]   3, 8, 0.50
  # Data = Data[,c('Y','X12', 'X2', 'X15', 'X4')]
  # Data = Data[,c('Y','X12', 'X2', 'X15', 'X4', 'X14')]
  

#################################

library(randomForest)

setwd('/Users/Kaveh/Documents/School Files/STAT 452/Assignments/Project 1')
source('Function Template.R')
Data = read.csv('Data2020.csv')

#################### LOAD DATA ####################

# Subset the data:
Data = Data[,c('Y','X12', 'X15', 'X4')]

# Tune the RF
all.mtry       = 3#1:12
all.nodesize   = 8#1:10
all.samplesize = 0.50#seq(from=0.5,to=0.75,by=0.05)

all.pars = expand.grid(mtry = all.mtry, nodesize = all.nodesize, ss = all.samplesize)
n.pars = nrow(all.pars)

M = 1
OOB.MSPEs = array(0, dim = c(M, n.pars))

min.OOB.MSPE = 10

for(i in 1:n.pars){
  
  print(paste0(i, " of ", n.pars))
  
  this.mtry = all.pars[i,"mtry"]
  this.nodesize = all.pars[i,"nodesize"]
  this.samplesize = all.pars[i,"ss"]
  
  for(j in 1:M){
    fit.rf = randomForest(Y ~ ., 
                          data = Data, 
                          importance = F,
                          mtry = this.mtry, 
                          nodesize = this.nodesize,
                          ntree = 1000)#,
                          #sampsize = this.samplesize*nrow(Data))
    
    OOB.pred = predict(fit.rf)
    OOB.MSPE = get.MSPE(Data$Y, OOB.pred)
    
    OOB.MSPEs[j, i] = OOB.MSPE 
    
    if(OOB.MSPE < min.OOB.MSPE) {
      min.OOB.MSPE = OOB.MSPE
      print(paste0("Minimum OOB MSPE - ", OOB.MSPE, " - acheived by parameters: (m, node, ss) --> ", 
                   this.mtry, ', ', 
                   this.nodesize, ', ', 
                   this.samplesize))
      print(fit.rf)
      
    } 
  }
}

# Get the samllest mean model
#sort(colMeans(OOB.MSPEs))

names.pars = paste0(all.pars$mtry,"-", all.pars$nodesize, "-", all.pars$ss)
colnames(OOB.MSPEs) = names.pars

### Make boxplot
#boxplot(OOB.MSPEs, las = 2, main = "MSPE Boxplot")

# RMSPE
OOB.RMSPEs = apply(OOB.MSPEs, 1, function(W) W/min(W))
OOB.RMSPEs = t(OOB.RMSPEs)
#boxplot(OOB.RMSPEs, las = 2, main = "RMSPE Boxplot")

sort(colMeans(OOB.MSPEs))
sort(colMeans(OOB.RMSPEs))






