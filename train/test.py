import torch
from sklearn.metrics import precision_score, recall_score

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
    
        y_true = []
        y_pred = []

        self.__model.eval()

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.__device)
                targets = targets.to(self.__device)

                outputs = self.__model(inputs)

                loss = self.__loss_fn(outputs, targets)

                samples += inputs.shape[0]
                cumulative_loss += loss.item() 
                _, predicted = outputs.max(1)

                correct_predictions += predicted.eq(targets).sum().item()

                y_true.extend(targets.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

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
        
        y_true = []
        y_pred = []

        self.__model.eval()

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.__device)
                targets = targets.to(self.__device)

                outputs = self.__model(inputs)
                _, predicted = outputs.max(1)

                y_true.extend(targets.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        
        return {"y_true": y_true, "y_pred" : y_pred}