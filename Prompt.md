# Prompt
Hey CHAT GPT, act as an application develpoer expert, in Python using streamlit, and build a machine learning 
application using scikit-learn with the following workflow:

1) Greet the User with a welcome message and a bried description of the application.
2) Ask the user if hw wants to upload the data or use the example data.
3) If the user select to upload the data, show the uploader selection on th sidebar, upload the dataset in
csv, tsv, xslx, or data simple formate.
4) If the user do not want to upload the data when provide a default datset selection box on the side.
    This selection box should download the data from sns.load_dataset() funcation. The datasets should include
     titanic, tips, or iris.
5) Print the basic information such as data head, data shape, data description and data info and coulmns name.
6)  Ask from the user to select the columns as feature and also columns as target.
7) Identify the problem if the columns is continuous numaric columns the print the message that this is a regression problem,
otherwise print the message this is a classification problem.
8) Pre-process the data , if the data contain any missing values, then fill the missing values with iterative impute funcation of scikit-learn , if the feature are not in the same scale ,then scale the features using the standard scaler funcation of scikit-learn. If the feature are categorical variable then encode the categorical variable using label encoder funcation of scikit-learn. Please keep it mind to, the encoder separate of each column as we nedd to inverse transform of the data at the end.
9) Ask the user to provide the train test split size via slider or user input funcation.
10) Ask the user to select the model from the sidebar, the model should include the Linear Regression, Decision Tree, random forest and support vector machine and same class of the models for the classification problems.
11) Train the models on the training data and Evalute on the test data.
12) If the problem is a regression problem, use the mean square error, RMSE, MAE, AUROC, curve and r2 score for evalution, 
if the problem is classification problem , use the accuracy score, precision , recall, f1 score, and draw confusion matrix for evalution.
13) print the evalution matrix of each model.
14) Highlight the best model on the base of evalution matrix.
15) Ask the user if he want to download the model, then if yes, download the model in the pickle format.
16) Ask the user if he want to make the precdiction , if yes then ask user to provide input data using slider, or uploaded the file, and make the prediction using the best model.
17) Show the prediction to the user. 

## Modification/ Fine Tunning of the model:
1) please modify and ask the user to provide the information if the problem is regression or classification, also, do not run anything untill we select the columns and tell the app to run analysis, please add one button strats training ml models.
2) Please also use cache for data and models to speedup the procedure.
 