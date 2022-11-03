import torch
from copy import deepcopy

'''
Remarks 

    1. The training loop ('training' method) focuses on training a medical task model. 
        This means that we are mainly interested in the recall scores of the unhealthy classes.
        For this purpose, the callback function is an Early Stopping technique focusing on 
        the improvement of the validation loss and avg recall of the unhealthy classes (min_delta = 0) 
        
    2. Per training epoch we see/print the progress of the loss, accuracy and recall metrics.
    
    3. The resulting best model is saved in a .pt file
    
    4. Labels should be in integers (not strings) starting from 0, ex. 0,1,2,3 for a 4-class problem
        If data is loaded via Dataloaders the above format is loaded automatically by the ImageFolder method.
        In order to verify the the class name-label assignment created by the Dataloader you may type
        "print(train_dataset.class_to_idx)"
        
    5. The 'training' method returns two dictionaries that contain the loss and metrics history 
        for the training and validation phases respectively.
        Each dictionary has the following self-explanatory keys: 
            - 'loss', 'accuracy', 'avg_recall', 'avg_precision', 'avg_f1'; 
                and the values are lists of the respective epoch values
            - 'recall_per_class', 'precision_per_class', 'f1_per_class';
                and the values are lists which consist of sublists equal to the number of classes.
                Each sublist describes the class metric history per epoch
'''

class Train():
    
    def __init__(self, model, loss_fct, optimizer, 
                 train_loader, validation_loader, no_of_classes, label_of_normal_class):
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('Device:', self.device)
        
        self.model = model                                 # nn model                
        self.loss_fct = loss_fct                           # loss function
        self.optimizer = optimizer                         # loss optimizer
        self.epochs = 100
        self.train_loader = train_loader                   
        self.validation_loader = validation_loader
        self.no_of_classes = no_of_classes
        self.label_of_normal_class = label_of_normal_class # see Remark 4 above
        self.patience = 20
        
        self.model.to(self.device)
        
    #------------------------------------------------------------------------------------------------------
    
    def training(self):
        
        # initialize empty dictionaries to save loss and metrics history PER EPOCH
        self.training_history = {'loss':[], 'accuracy':[], 'avg_recall':[], 'avg_precision':[], 'avg_f1':[], 
            'recall_per_class':[[] for _ in range(self.no_of_classes)], 
            'precision_per_class':[[] for _ in range(self.no_of_classes)], 
            'f1_per_class':[[] for _ in range(self.no_of_classes)]}
        self.validation_history = {'loss':[], 'accuracy':[], 'avg_recall':[], 'avg_precision':[], 'avg_f1':[], 
            'recall_per_class':[[] for _ in range(self.no_of_classes)], 
            'precision_per_class':[[] for _ in range(self.no_of_classes)],
            'f1_per_class':[[] for _ in range(self.no_of_classes)]}

        # below block is used in the Early Stopping section of the loop
        self.threshold_val_loss    = 10e+5
        self.threshold_avg_recall  = 0
        self.best_model            = deepcopy(self.model)
        self.unchanged_epochs      = 0
        
        print('Starting training...')
        
        for e in range(0, self.epochs):

            print('-'*25)
            print(f'Epoch {e + 1}/{self.epochs}')

            #------Training-----------------------------------
            self.model.train() # set model to training phase
            
            print('-->Training phase')
            
            #train_epoch
            self.train_epoch()
            
            #epoch_metrics
            self.epoch_metrics(self.training_history)

            #-------Validation---------------------------------
            self.model.eval() # set model to validation phase
            
            print('-->Validation phase')
            
            #validate_epoch
            self.validate_epoch()
            
            #epoch_metrics
            self.epoch_metrics(self.validation_history)
            
            #-------Early_Stopping------------------------------
            self.early_stopping_check()
            
            if self.unchanged_epochs == self.patience:
                break
        
        print('Training complete!')
        return self.training_history, self.validation_history
    
    #------------------------------------------------------------------------------------------------------
    
    def train_epoch(self):
        
        train_loss = 0
        self.target_true = [0 for _ in range(self.no_of_classes)]; 
        self.predicted_true = [0 for _ in range(self.no_of_classes)]; 
        self.correct_true = [0 for _ in range(self.no_of_classes)]
    
        for train_step, (images, labels) in enumerate(self.train_loader): 
            images, labels = images.to(self.device), labels.to(self.device)  #send data to device 
    
            # batch training
            self.optimizer.zero_grad()                      #zero the parameter gradients
            outputs = self.model(images)                    #forward
            loss = self.loss_fct(outputs, labels)           #compute loss
            loss.backward()                                 #backward
            self.optimizer.step()                           #optimize
            train_loss += loss.item()                       #add batch loss to the total epoch train loss
    
            # class metrics - batch level
            labels_list = torch.Tensor.tolist(labels)
            _, preds = torch.max(outputs, 1)            
            preds_list = torch.Tensor.tolist(preds)
            self.batch_metrics(labels_list, preds_list)
            
        # mean epoch train loss
        train_loss /= (train_step + 1)
        print(f'Loss={train_loss:.4f}')
        self.training_history['loss'].append(train_loss)
     
    #------------------------------------------------------------------------------------------------------   
     
    def validate_epoch(self):
        
        val_loss = 0
        self.target_true = [0 for _ in range(self.no_of_classes)]; 
        self.predicted_true = [0 for _ in range(self.no_of_classes)]; 
        self.correct_true = [0 for _ in range(self.no_of_classes)]
        
        with torch.no_grad():
            for val_step, (images, labels) in enumerate(self.validation_loader):
                images, labels = images.to(self.device), labels.to(self.device)   #send data to device
        
                # batch validation
                outputs = self.model(images)             #forward
                loss = self.loss_fct(outputs, labels)    #compute loss
                val_loss += loss.item()                  #add batch loss to the total epoch validation loss
        
                # class metrics
                labels_list = torch.Tensor.tolist(labels)
                _, preds = torch.max(outputs, 1)
                preds_list = torch.Tensor.tolist(preds)
                self.batch_metrics(labels_list, preds_list)
            
        # mean epoch validation loss
        val_loss /= (val_step + 1)
        print(f'Loss={val_loss:.4f}')
        self.validation_history['loss'].append(val_loss)

    #------------------------------------------------------------------------------------------------------
    
    def batch_metrics(self, labels_list, preds_list):

        for clas in range(self.no_of_classes):
            # no of real class points
            self.target_true[clas] += len([i for i in labels_list if i==clas])
            # no of predicted class points
            self.predicted_true[clas] += len([i for i in preds_list if i==clas]) 
            # no of correctly predicted class points
            self.correct_true[clas] += len([i for i, (x, y) in enumerate(zip(labels_list, preds_list)) if x == y and x==clas])
            
    #------------------------------------------------------------------------------------------------------
    
    def epoch_metrics(self, dictionary):
    
        # CLASS METRICS on epoch level
        recalls = [0 for _ in range(self.no_of_classes)]; 
        precisions = [0 for _ in range(self.no_of_classes)]; 
        f1_scores = [0 for _ in range(self.no_of_classes)]
        
        for clas in range(self.no_of_classes):
            
            recalls[clas] = round(self.correct_true[clas] / self.target_true[clas], 2)  
            dictionary['recall_per_class'][clas].append(recalls[clas])
            
            if self.predicted_true[clas]==0:
                precisions[clas]=0
            else:
                precisions[clas] = round(self.correct_true[clas] / self.predicted_true[clas], 2)                              
            dictionary['precision_per_class'][clas].append(precisions[clas])
            
            f1_scores[clas] = 2 * round(precisions[clas] * recalls[clas] / (precisions[clas] + recalls[clas]), 2) 
            dictionary['f1_per_class'][clas].append(f1_scores[clas])
    
        # MACRO AVG METRICS on epoch level
        accuracy = round(sum(self.correct_true)/sum(self.target_true), 2)
        dictionary['accuracy'].append(accuracy)
        
        recall = round(sum(recalls)/len(recalls), 2)
        dictionary['avg_recall'].append(recall)
        
        if len(precisions)>0:
            precision = round(sum(precisions)/len(precisions), 2)
        else:
            precision = 0                       
        dictionary['avg_precision'].append(precision)
            
        f1_score = round(sum(f1_scores)/len(f1_scores), 2)
        dictionary['avg_f1'].append(f1_score)
    
        print(f'Accuracy={accuracy} - Recall per class={recalls}')
        
    #------------------------------------------------------------------------------------------------------
    def early_stopping_check(self):
        
        #current epoch validation loss and avg recall of all disease classes 
        epoch_val_loss = self.validation_history['loss'][-1]
        epoch_val_avg_recall = sum([j[-1] for (i, j) in enumerate(self.validation_history['recall_per_class']) if i!=self.label_of_normal_class]) \
                                / (self.no_of_classes - 1)
        
        # αν το loss πεφτει και αν τα unhealthy class recalls αυξηθηκαν on avg όρισε νέο best_model και ανανεώσε τα thresholds
        if (epoch_val_loss < self.threshold_val_loss) and (self.threshold_avg_recall < epoch_val_avg_recall):      
            
            del self.best_model                                  # διεγραψε το παλιο best model
            self.best_model = deepcopy(self.model)               # θεσε ως best το νεο
            
            self.unchanged_epochs = 0                            # epoch counter ξανα στο 0
            self.threshold_val_loss  = epoch_val_loss            # θεσε το νέο loss ως μέγιστο target
            self.threshold_avg_recall = epoch_val_avg_recall     # θεσε το νέο avg recall ως ελάχιστο target
            
            
            current_epoch = len(self.validation_history['loss'])
            torch.save(self.best_model, f'model_epoch{current_epoch}.pt') 
            
            print('-->New model saved!')
            
        else:
            self.unchanged_epochs += 1
