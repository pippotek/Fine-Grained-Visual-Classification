import torch
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class Tester:
    def __init__(self, model, dataloaders, device, loss_fn):
        self.__model = model
        self.__data_loaders = dataloaders
        self.__device = device
        self.__loss_fn = loss_fn

    def test_step(self, test=False, eval=False, train=False, precision=False, recall=False):
        
        assert test + eval + train == 1, "Exactly one of test, eval, or train must be True"

        if test:
            assert isinstance(test, bool), "test must be a boolean"
            data_loader = self.__data_loaders["test_loader"]
        elif eval:                    
            assert isinstance(eval, bool), "test must be a boolean"
            data_loader = self.__data_loaders["val_loader"]
        elif train:
            assert isinstance(train, bool), "test must be a boolean"
            data_loader = self.__data_loaders["train_loader"]            
        else:
            raise ValueError("One of test, eval or train must be True")

        samples = 0.
        cumulative_loss = 0.
        correct_predictions = 0

        num_samples = len(data_loader.dataset)
        y_true = np.zeros(num_samples, dtype=int)
        y_pred = np.zeros(num_samples, dtype=int)

        self.__model.eval()

        with torch.no_grad():
            index = 0
            for inputs, targets in data_loader:
                inputs = inputs.to(self.__device)
                targets = targets.to(self.__device)

                outputs = self.__model(inputs)

                loss = self.__loss_fn(outputs, targets)

                batch_size = inputs.shape[0]
                samples += inputs.shape[0]
                cumulative_loss += loss.item() 
                _, predicted = outputs.max(1)

                correct_predictions += predicted.eq(targets).sum().item()
                
                y_true[index:index + batch_size] = targets.cpu().numpy()
                y_pred[index:index + batch_size] = predicted.cpu().numpy()
                index += batch_size

        average_loss = cumulative_loss / samples
        accuracy = correct_predictions / samples * 100

        if precision:
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100
        if recall:
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100
        
        return average_loss, accuracy, precision, recall
    
    def get_predictions(self, test=False, train=False, eval=False):

        assert test + eval + train == 1, "Exactly one of test, eval, or train must be True"

        if test:
            assert isinstance(test, bool), "test must be a boolean"
            data_loader = self.__data_loaders["test_loader"]
        elif eval:                    
            assert isinstance(eval, bool), "test must be a boolean"
            data_loader = self.__data_loaders["val_loader"]
        elif train:
            assert isinstance(train, bool), "test must be a boolean"
            data_loader = self.__data_loaders["train_loader"]            
        else:
            raise ValueError("One of test, eval or train must be True")
        
        num_samples = len(data_loader.dataset)

        # Preallocate numpy arrays
        y_true = np.zeros(num_samples, dtype=int)
        y_pred = np.zeros(num_samples, dtype=int)

        self.__model.eval()

        with torch.no_grad():
            index = 0
            for inputs, targets in data_loader:
                inputs = inputs.to(self.__device)
                targets = targets.to(self.__device)

                batch_size = inputs.shape[0]
                outputs = self.__model(inputs)
                _, predicted = outputs.max(1)

                y_true[index:index + batch_size] = targets.cpu().numpy()
                y_pred[index:index + batch_size] = predicted.cpu().numpy()
                index += batch_size
        
        return {"y_true": y_true, "y_pred" : y_pred}
    
    def plot_confusion_matrix(self, return_confusion_matrix=False, width=10, height=8, cmap="Blues", fmt="d"):
        
        dict_preds = self.get_predictions(test=True)

        y_true = dict_preds["y_true"]
        y_pred = dict_preds["y_pred"]

        conf_matrix = confusion_matrix(y_true, y_pred)
        classes = np.unique(y_pred)
        return classes, conf_matrix

        plt.figure(figsize=(width, height)) 
        plot = sns.heatmap(confusion_matrix, annot=True, fmt=fmt, cmap=cmap,
                            xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
        
        if return_confusion_matrix:
            return conf_matrix