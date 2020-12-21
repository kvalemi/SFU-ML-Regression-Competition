## Machine Learning Classification Competition

My upper-level STAT 452 organized a machine learning competition that focused on an anonymized regression problem. The dataset was completely anonymized meaning that I had no knowledge of what the features and response actually represented. By doing this, more time had to be spent on tuning the prediction machine rather than researching the domain of the problem.

## Models Trained

- MLR Least Squares
- Stepwise (AIC)
- Stepwise (BIC)
- Ridge Regression
- LASSO Least Squares
- GAM (Fitted Splines)
- Random Forest


## Steps Taken

1) 
LS, Stepwise, Ridge Regression, LASSO, and GAM were iterated over a 10-fold cross validation and an MSPE and RMSPE were calculated from each validation fold. Random Forest was tuned on a 720 parameter-combination grid search and an OOB MSPE and RMSPE were calculated from each OOB sample. I then derived the mean and standard deviation of each model’s MSPE and RMSPE and used these metrics, respectively, to assess the bias-variance trade-off of each model. I looked for accuracy and consistency amongst these metrics and decided to proceed with Random Forest due to its stellar performance.

2) 
After choosing Random Forest, I decided to only tune and optimize this model. Initially, I performed a grid search on the following parameters: 

-	mtry = (1:12) 
-	Node Size = (1:10)
-	Sample Size = (0.50, 0.55, …, 0.75) 

After running this grid search, I made several observations from the OOB RMSPE’s. The top ten Random Forest’s with the smallest mean OOB RMSPEs all had the following in common:

-	They were all using a Sample Size parameter of 0.50
-	They had an mtry parameter ranging from 3 to 6
-	They had Node Size parameters ranging from 1 to 5

I refined the grid search to only iterate over the above ranges. After obtaining results, I noticed that the top model, with the smallest mean OOB RMSPEs, had mtry = 3 and Node Size = 8. 

3) 
Having tuned my parameters sufficiently, I decided to put the rest of my effort into feature selection. I ran the Random Forest function varImPlot() on the raw data to assess the importance of each feature. I then chose the top 7 most important features and retrained my optimal Random Forest only on these features. To further investigate these 7 features, I decided to run my optimal model on every possible subset of these features and chose the subset that resulted in the smallest mean OOB RMSPE. After all of this tuning, I arrived at my final model being a Random Forest with parameters mtry = 3, Node Size = 8, Sample Size = 0.50 while only utilizing the features X12, X15, and X4.

## Final Model

Final Model:

![](/Final%20Model.png)
