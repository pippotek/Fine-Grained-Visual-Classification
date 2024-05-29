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

                if not self.model.__class.name == "Network_Wrapper":
                    outputs = self.__model(inputs)
                else:
                    _, _, _, outputs, _, _, _ = self.__model(inputs)

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
                if not self.__model.__class__.__name__ == "Network_Wrapper":                
                    outputs = self.__model(inputs)
                    _, predicted = outputs.max(1)
                else:
                    netp = torch.nn.DataParallel(self.__model, device_ids=[0])
                    _, _, _, outputs, _, _, _ = netp(inputs)
                    _, predicted = torch.max(outputs.data, 1)

                y_true[index:index + batch_size] = targets.cpu().numpy()
                y_pred[index:index + batch_size] = predicted.cpu().numpy()
                index += batch_size
        
        return {"y_true": y_true, "y_pred" : y_pred}
    
    def top_k_accuracy(self, train=False, eval=False, test=False, k=5):
        
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
        
        self.__model.eval()
        top_k_correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.__device)
                targets = targets.to(self.__device)

                if not self.__model.__class__.__name__ == "Network_Wrapper":                
                    outputs = self.__model(inputs)
                else:
                    netp = torch.nn.DataParallel(self.__model, device_ids=[0])
                    _, _, _, outputs, _, _, _ = netp(inputs)
                
                _, top_k_preds = outputs.topk(k, dim=1, largest=True, sorted=True)
                
                # Check if the true labels are in the top k predictions
                correct = top_k_preds.eq(targets.view(-1, 1).expand_as(top_k_preds))
                
                # Sum the number of correct predictions
                top_k_correct += correct.sum().item()
                total += targets.size(0)
        
        top_k_accuracy = top_k_correct / total
        return top_k_accuracy