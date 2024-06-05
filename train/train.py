import os
os.chdir("/home/filippo/Desktop/Uni/ML/Fine-Grained-Visual-Classification") # to import modules from other directories
print("Warning: the working directory was changed to", os.getcwd())

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score
import numpy as np
from train.test import Tester

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models_methods')))

# Import the cmal_train function
from methods.CMAL.builder_resnet import cmal_train

class Trainer(Tester):

    def __init__(self, 
                 data_loaders: dict, 
                 dataset_name: str,
                 model: torch.nn.Module, 
                 optimizer: torch.optim, 
                 loss_fn: torch.nn, 
                 device, 
                 seed: int,
                 exp_name, # the name of this experiment
                 exp_path, # where you keep all the experiments
                 use_early_stopping=True,
                 patience=5,
                 delta=1e-3,
                 scheduler=None,
                 num_classes = None):
        """
        The exp_name should be a string containing all the information about the experiment:
        - model 
        - optimizer 
        - loss function
        - other hyperparameters 
        """
        super().__init__(model, data_loaders, device, loss_fn, num_classes)
        self.__optimizer = optimizer        
        self.__use_early_stopping = use_early_stopping
        self.__seed = seed
        self.__epoch = 0
        self.__trained = False
        self.__scheduler = scheduler
        self.__best_loss = float("inf")

        assert os.path.exists(exp_path), "Experiment path does not exist"
        assert os.path.exists(os.path.join(exp_path, exp_name)) == False, "The experiment already exists"
        assert isinstance(patience, int) and patience > 0, "patience must be a positive integer"
        assert isinstance(delta, float) and delta > 0, "delta must be a positive float"
        assert isinstance(use_early_stopping, bool), "use_early_stopping must be a boolean"
        assert isinstance(data_loaders, dict), "data_loaders must be a dictionary with keys 'train_loader', 'val_loader', 'test_loader'"

        self.__exp_name = os.path.join(exp_path, exp_name)
        
        sam_schedulers = ["StepLR_SAM"]
        if self.__optimizer.__class__.__name__ == "SAM": 
            from torch.nn import CrossEntropyLoss
            assert loss_fn.label_smoothing >= 0.07, "smoothing must be >= 0.7 when using SAM"
            assert isinstance(loss_fn, CrossEntropyLoss), "loss function must be CrossEntropyLoss with label_smoothing when using SAM"   
            if self.__scheduler is not None:
                assert self.__scheduler.__class__.__name__ in sam_schedulers, f"Torch schedulers don't work with SAM. You must use any of {sam_schedulers}"
            rho_SAM = self.__optimizer.param_groups[0]['rho']  
        else:
            rho_SAM = None

        if self.__scheduler is not None:
            if self.__scheduler.__class__.__name__ in sam_schedulers:
                assert self.__optimizer.__class__.__name__ == "SAM", "Use a torch scheduler if you are not using SAM"      

        if self.__use_early_stopping:
            from utility.early_stopping import EarlyStopping
            self.__early_stopping = EarlyStopping(patience=patience, 
                                                  delta=delta,
                                                  path=os.path.join(self.__exp_name,"checkpoint.pth"))      

        os.makedirs(self.__exp_name, exist_ok=True) 
        self.__writer = SummaryWriter(log_dir=f"{self.__exp_name}")
        self.__save_config(dataset_name, rho_SAM, patience, delta)

    def get_model(self):
        return self._Tester__model
    
    def get_optimizer(self):
        return self.__optimizer
    
    def get_scheduler(self):
        return self.__scheduler
    
    def get_device(self):
        return self._Tester__device
    
    def get_exp_name(self):
        return self.__exp_name

    def __train_step(self, verbose, log_interval):
        """
        log_interval can be an integer or a float between 0 and 1. If it is an integer, the function will print the statistics every log_interval steps.
        If it is a float, the function will print the statistics every log_interval*num_of_batches steps.
        """
        samples = 0.0
        cumulative_loss = 0.0
        cumulative_accuracy = 0.0
        
        assert isinstance(verbose, bool), "verbose must be a boolean"
        assert isinstance(log_interval, (int, float)) and log_interval>0, "log_interval must be an integer or a float and non-negative"
        if log_interval < 1:
            log_interval = int(len(self._Tester__data_loaders["train_loader"])*log_interval)

        self._Tester__model.train()

        if self.__optimizer.__class__.__name__ == "SAM":
            from utility.bypass_bn import enable_running_stats, disable_running_stats

        num_samples = len(self._Tester__data_loaders["train_loader"].dataset)
        y_true = np.zeros(num_samples, dtype=int)
        y_pred = np.zeros(num_samples, dtype=int)

        index = 0
        for batch_idx, (inputs, targets) in enumerate(self._Tester__data_loaders["train_loader"]):
            inputs, targets = inputs.to(self._Tester__device), targets.to(self._Tester__device)
            
            # if inputs.shape[0] < batch_size:  # check that number of input samples is less than batch size
            # continue

            if not self._Tester__model.__class__.__name__ == "Network_Wrapper":
                # first forward-backward step
                if self.__optimizer.__class__.__name__ == "SAM":        
                    enable_running_stats(self._Tester__model) # disable batch norm running stats

                outputs = self._Tester__model(inputs)

                if isinstance(outputs, dict): # used for PIM

                    loss = 0 # initialize loss

                    for name in outputs:
                    
                        if "drop_" in name:
                            S = outputs[name].size(1)
                            logit = outputs[name].view(-1, self._Tester__num_classes).contiguous()
                            n_preds = torch.nn.Tanh()(logit)
                            labels_0 = torch.zeros(n_preds.size()) - 1
                            labels_0 = labels_0.to(self._Tester__device)
                            loss_n = torch.nn.MSELoss()(n_preds, labels_0)
                            loss += 5 * loss_n

                        elif "layer" in name:    
                            loss_b = torch.nn.CrossEntropyLoss()(outputs[name].mean(1), targets)
                            loss += 0.5* loss_b
                        
                        elif "comb_outs" in name:
                            loss_c = torch.nn.CrossEntropyLoss()(outputs[name], targets)
                            loss += 1 * loss_c
                else:
                    loss = self._Tester__loss_fn(outputs, targets)
                
                loss.backward()
                
                if self.__optimizer.__class__.__name__ == "SAM":
                    self.__optimizer.first_step(zero_grad=True)
                    # second forward-backward step
                    disable_running_stats(self._Tester__model)
                    loss = self._Tester__loss_fn(self._Tester__model(inputs), targets)
                    loss.backward()
                    self.__optimizer.second_step(zero_grad=True)
                else:
                    self.__optimizer.step()
                    self.__optimizer.zero_grad()

                cumulative_loss += loss.item()
                
                if isinstance(outputs, dict):
                    predicted = torch.argmax(outputs['comb_outs'][0])
                else:
                    _, predicted = outputs.max(dim=1)
                cumulative_accuracy += predicted.eq(targets).sum().item()

            else:
                cumulative_loss, batch_accuracy, predicted = cmal_train(
                    inputs=inputs, targets=targets, net=self._Tester__model,
                    optimizer=self.__optimizer, loss=self._Tester__loss_fn,
                    scheduler=self.__scheduler
                )
                cumulative_accuracy += batch_accuracy

            samples += inputs.shape[0]

            if verbose and batch_idx % log_interval == 0:
                current_loss = cumulative_loss / samples
                current_accuracy = cumulative_accuracy / samples * 100
                print(f'Batch {batch_idx}/{len(self._Tester__data_loaders["train_loader"])}, Loss: {current_loss:.4f}, Accuracy: {current_accuracy:.2f}%', end='\r')
            
            batch_size = inputs.shape[0]
            y_true[index:index + batch_size] = targets.cpu().numpy()
            y_pred[index:index + batch_size] = predicted.cpu().numpy()
            index += batch_size

        if self.__scheduler:
            self.__scheduler.step()

        accuracy = cumulative_accuracy / samples * 100    
        avg_loss = cumulative_loss / samples    
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100

        return avg_loss, accuracy, precision, recall

    def main(self,
             epochs=10,
             verbose_steps=True, # print after log_interval-learning steps
             log_interval=10): 

        from utility.initialize import initialize
        initialize(self.__seed)
            
        self._Tester__model.to(self._Tester__device)
                
        # Log to TensorBoard
        if self.__trained == False:
            self.__trained = True
            print("Before training:")
            train_loss, train_accuracy, train_precision, train_recall = self.test_step(train=True, precision=True, recall=True)
            val_loss, val_accuracy, val_precision, val_recall = self.test_step(eval=True,precision=True, recall=True) 
            test_loss, test_accuracy, test_precision, test_recall = self.test_step(test=True,precision=True, recall=True)
            self.__log_values(self.__writer, self.__epoch, train_loss, train_accuracy,train_precision, train_recall, "Train")
            self.__log_values(self.__writer, self.__epoch, val_loss, val_accuracy, val_precision, val_recall, "Validation")
            self.__log_values(self.__writer, self.__epoch, test_loss, test_accuracy, test_precision, test_recall, "Test")
            self.__print_statistics(train_loss, train_accuracy, val_loss, val_accuracy, test_loss, test_accuracy)
        
        pbar = tqdm(range(epochs), desc="Training")
        for _ in pbar:
            train_loss, train_accuracy, train_precision, train_recall = self.__train_step(verbose=verbose_steps, log_interval=log_interval)
            val_loss, val_accuracy, val_precision, val_recall = self.test_step(eval=True) 

            print("-----------------------------------------------------")
            self.__epoch += 1
            self.__log_values(self.__writer, self.__epoch, train_loss, train_accuracy,train_precision, train_recall, "Train")
            self.__log_values(self.__writer, self.__epoch, val_loss, val_accuracy, val_precision, val_recall, "Validation")

            pbar.set_postfix(train_loss=train_loss, train_accuracy=train_accuracy, val_loss=val_loss, val_accuracy=val_accuracy)

            if self.__use_early_stopping:
                self.__early_stopping(val_loss,self._Tester__model, self.__optimizer, self.__scheduler)
                if self.__early_stopping.early_stop:
                    print("Early stopping")
                    break       
            else:
                if val_loss < self.__best_loss:
                    self.__best_loss = val_loss
                    torch.save({
                        "model": self._Tester__model.state_dict(),
                        "optimizer": self.__optimizer.state_dict(),
                        "scheduler": self.__scheduler.state_dict() if self.__scheduler is not None else None
                        }, 
                        os.path.join(self.__exp_name,"checkpoint.pth"))
        
        # Compute final evaluation results
        print("After training:")
        train_loss, train_accuracy, train_precision, train_recall = self.test_step(train=True, precision=True, recall=True)
        val_loss, val_accuracy, val_precision, val_recall = self.test_step(eval=True, precision=True, recall=True) 
        test_loss, test_accuracy, test_precision, test_recall = self.test_step(test=True, precision=True, recall=True)

        self.__log_values(self.__writer, self.__epoch, train_loss, train_accuracy,train_precision, train_recall, "Train")
        self.__log_values(self.__writer, self.__epoch, val_loss, val_accuracy, val_precision, val_recall, "Validation")
        self.__log_values(self.__writer, self.__epoch, test_loss, test_accuracy, test_precision, test_recall, "Test")

        self.__print_statistics(train_loss, train_accuracy, val_loss, val_accuracy, test_loss, test_accuracy)

        # Flush the logs to disk 
        self.__writer.flush()            

    def close_writer(self):
        self.__writer.close()
        print("Writer closed")

    def open_writer(self):
        self.__writer = SummaryWriter(log_dir=f"{self.__exp_name}")
        print("A new writer was opened opened")

    def set_exp_name(self, new_name):
        self.__exp_name = new_name
        self.__writer = SummaryWriter(log_dir=f"{self.__exp_name}")
        print(f"Experiment name was changed to {new_name}")
    
    def __print_statistics(self, train_loss, train_accuracy, val_loss, val_accuracy, test_loss, test_accuracy):
        print(f"\tTraining loss {train_loss:.5f}, Training accuracy {train_accuracy:.2f}")
        print(f"\tValidation loss {val_loss:.5f}, Validation accuracy {val_accuracy:.2f}")
        print(f"\tTest loss {test_loss:.5f}, Test accuracy {test_accuracy:.2f}")
        print("-----------------------------------------------------")

    # tensorboard logging utilities
    def __log_values(self, writer, step, loss, accuracy,precision, recall, prefix):
        writer.add_scalar(f"{prefix}/loss", loss, step)
        writer.add_scalar(f"{prefix}/accuracy", accuracy, step)
        writer.add_scalar(f"{prefix}/precision", precision, step)
        writer.add_scalar(f"{prefix}/recall", recall, step)
        
    def __save_config(self, dataset_name, rho_SAM, patience, delta):
        config = {
            'data': { 
                'batch_size':self._Tester__data_loaders["train_loader"].batch_size,
                'dataset_name': dataset_name
            }, 
            'model': self._Tester__model.__class__.__name__ if not self._Tester__model.__class__.__name__ == "Network_Wrapper" else "CMAL",
            'optimizer': {
                'optimizer': self.__optimizer.__class__.__name__,
                'momentum': self.__optimizer.param_groups[0]['momentum'],
                'weight_decay': self.__optimizer.param_groups[0]['weight_decay'],
                'lr': self.__optimizer.param_groups[0]['lr'],
                "rho_SAM": rho_SAM 
            } ,
            'scheduler': self.__scheduler.__class__.__name__ if self.__scheduler is not None else None,
            'loss_fn': {
                'loss_fn': self._Tester__loss_fn.__class__.__name__,
                'smoothing': self._Tester__loss_fn.label_smoothing
            },
            'seed': self.__seed,
            'early_stopping': {
                'use_early_stopping': self.__use_early_stopping,
                'patience': patience,
                'delta': delta
            }
        }
        import json
        config_file_path = f"{self.__exp_name}/config.json"
        with open(config_file_path, 'w') as file:
            json.dump(config, file, indent=4)