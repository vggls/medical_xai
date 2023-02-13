'''
Sources : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
          https://plotly.com/python/roc-and-pr-curves/
'''

import numpy as np
import plotly.graph_objects as go
import pandas as pd
from sklearn.metrics import classification_report, balanced_accuracy_score, accuracy_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
import torch
import torch.nn as nn

class Test_Report():
    
    def __init__(self, dataloader, model, classes):
        
        # ------------------- set device attribute -----------------------------------------------
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('Device:', self.device)
        
        # ------------------- main attributes ---------------------------------------------------
        self.dataloader = dataloader
        self.model = model
        self.classes = classes   # list of strings

        self.model.to(self.device)

        # ------ attributes to be used for calculations in the class methods ---------------------
        self.y_true_labels = []; self.y_pred_labels = []
        scores = [];
        
        softmax = nn.Softmax(dim=1)
        for batch, batch_labels in self.dataloader:
            batch, batch_labels = batch.to(self.device), batch_labels.to(self.device)
            
            batch_prediction = self.model(batch)
            scores += softmax(batch_prediction).tolist()
            self.y_pred_labels += torch.argmax(batch_prediction, dim = 1).tolist()
            
            self.y_true_labels += batch_labels.tolist()
        
        # Model scores organized in an np.array. Each row corresponds to a datapoint and is a prob distribution. 
        self.y_scores = np.array(scores)
        
        # One hot encode the labels in order to plot them
        self.y_onehot = pd.get_dummies(self.y_true_labels)
        self.y_onehot.columns = self.classes
        
        # -----------------end of attributes-------------------------------------------------------
    
    
    def classification_report(self):
            
        print(classification_report(y_true = self.y_true_labels, 
                                    y_pred = self.y_pred_labels,
                                    target_names = self.classes))
    
    def balanced_accuracy(self):
                
        return balanced_accuracy_score(self.y_true_labels, self.y_pred_labels)

    def accuracy(self):
        
        return accuracy_score(self.y_true_labels, self.y_pred_labels)
    
    def f1(self, avg):
        
        return f1_score(self.y_true_labels, self.y_pred_labels, average=avg)

    def roc_curve_and_scores(self, plot=True):
        
        all_auc_scores = []

        fig = go.Figure()
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        
        for i in range(self.y_scores.shape[1]):
            y_true = self.y_onehot.iloc[:, i]
            y_score = self.y_scores[:, i]
        
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_score = roc_auc_score(y_true, y_score)
            all_auc_scores.append(auc_score)

            if plot:
              name = f"{self.y_onehot.columns[i]} (AUC={auc_score:.2f})"
              fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

        if plot:
          fig.update_layout(
              title_text='ROC curve', 
              title_x=0.35,
              xaxis_title='False Positive Rate',
              yaxis_title='True Positive Rate',
              yaxis=dict(scaleanchor="x", scaleratio=1),
              xaxis=dict(constrain='domain'),
              width=650, height=400
          )
          fig.show()

        return all_auc_scores

    def pr_curve_and_scores(self, plot=True):

        all_auc_scores = []

        fig = go.Figure()
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=1, y1=0
        )
        
        for i in range(self.y_scores.shape[1]):
            y_true = self.y_onehot.iloc[:, i]
            y_score = self.y_scores[:, i]
        
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            auc_score = average_precision_score(y_true, y_score)
            all_auc_scores.append(auc_score)

            if plot:
              name = f"{self.y_onehot.columns[i]} (AP={auc_score:.2f})"
              fig.add_trace(go.Scatter(x=recall, y=precision, name=name, mode='lines'))
        
        if plot:
          fig.update_layout(
              title_text='PR curve', 
              title_x=0.35,
              xaxis_title='Recall',
              yaxis_title='Precision',
              yaxis=dict(scaleanchor="x", scaleratio=1),
              xaxis=dict(constrain='domain'),
              width=650, height=400
          )
          fig.show()

        return all_auc_scores
            