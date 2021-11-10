import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score

from sklearn.model_selection import train_test_split


import mlflow
import mlflow.sklearn




if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file (make sure you're running this from the root of MLflow!)
    cust_app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cust_data_processed.csv")
    data = pd.read_csv(cust_app_path)    
    

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["loan_approval_status"], axis=1)
    test_x = test.drop(["loan_approval_status"], axis=1)
    train_y = train[["loan_approval_status"]]
    test_y = test[["loan_approval_status"]]

    #C = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    #l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        
        model = LogisticRegression()
        model.fit(train_x, train_y)
        predictions =  model.predict(test_x)
        
    
        test_accuracy = accuracy_score(test_y, predictions)
        test_precision_score = precision_score(test_y, predictions)
        test_recall_score = recall_score(test_y, predictions)
        test_f1_score = f1_score(test_y, predictions)
        auc_score = roc_auc_score(test_y, predictions)
        metrics = {"Test_accuracy": test_accuracy, "Test_precision_score": test_precision_score,
                   "Test_recall_score":test_recall_score,"Test_f1_score":test_f1_score, "auc score":auc_score}
    
  # Log the value of the metric from this run.
        mlflow.log_metrics(metrics) 
       
        mlflow.sklearn.log_model(model, "LR-tuned parameters-wo autolog")
        
        

        
