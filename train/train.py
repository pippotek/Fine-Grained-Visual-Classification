import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pickle 
import warnings

import os
os.chdir("/home/peppe/01_Study/01_University/Semester/2/Intro_to_ML/Project/Code") # to import modules from other directories
print("Warning: the working directory was changed to", os.getcwd())

class Trainer:

    def __init__(self, 
                 data_loaders: dict, 
                 dataset_name: str,
                 model: torch.nn.Module, 
                 optimizer: callable, 
                 loss_fn: torch.nn, 
                 device, 
                 seed: int, 
                 exp_name, # the name of this experiment
                 exp_path, # where you keep all the experiments
                 use_SAM=False, 
                 weight_decay = 0.0005,
                 lr=0.1,
                 momentum=0.9, 
                 rho_SAM=2, 
                 use_early_stopping=True,
                 patience=5,
                 delta=1e-3,
                 scheduler=None):
        """
        The exp_name should be a string containing all the information about the experiment:
        - model 
        - optimizer 
        - loss function
        - other hyperparameters 
        """
        self.__data_loaders = data_loaders
        self.__model = model 
        self.__optimizer = optimizer
        self.__loss_fn = loss_fn
        self.__device = device
        self.__use_SAM = use_SAM
        self.__use_early_stopping = use_early_stopping
        self.__seed = seed
        self.__train_loss = 10000
        self.__test_loss = 10000
        self.__val_loss = 10000
        self.__train_accuracy = 0
        self.__test_accuracy = 0
        self.__val_accuracy = 0
        self.__epoch = 0
        self.__trained = False
        self.__scheduler = scheduler

        assert os.path.exists(exp_path), "Experiment path does not exist"
        assert os.path.exists(os.path.join(exp_path, exp_name)) == False, "The experiment already exists"
        
        os.makedirs(os.path.join(exp_path, exp_name), exist_ok=True) 
        self.__exp_name = os.path.join(exp_path, exp_name)
        assert isinstance(data_loaders, dict), "data_loaders must be a dictionary with keys 'train_loader', 'val_loader', 'test_loader'"
        self.__writer = SummaryWriter(log_dir=f"{self.__exp_name}")

        if self.__use_SAM: 
            from torch.nn import CrossEntropyLoss
            from models_methods.methods.SAM.sam import SAM

            assert self.__loss_fn.label_smoothing >= 0.07, "smoothing must be >= 0.7 when using SAM"
            assert isinstance(self.__loss_fn, CrossEntropyLoss), "loss function must be CrossEntropyLoss with label_smoothing when using SAM"   
            self.__optimizer = SAM(self.__model.parameters(), 
                                   self.__optimizer, 
                                   rho=rho_SAM, 
                                   adaptive=True, 
                                   lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            self.__optimizer = optimizer(self.__model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        if self.__use_early_stopping:
            from models_methods.utility.early_stopping import EarlyStopping
            self.__early_stopping = EarlyStopping(patience=patience, 
                                                  delta=delta,
                                                  path=f"{self.__exp_name + '/checkpoint.pt'}")
        self.__save_config(dataset_name, rho_SAM, momentum, weight_decay, lr, patience, delta)

    def get_model(self):
        return self.__model 
    
    def get_data(self):
        return self.__data_loaders
    
    def get_optimizer(self):
        return self.__optimizer
    
    def get_loss(self):
        return self.__loss_fn
    
    def get_device(self):
        return self.__device
    
    def use_SAM(self):
        return self.__use_SAM
    
    def get_exp_name(self):
        return self.__exp_name

    def __train_step(self, verbose, log_interval):
        samples = 0.0
        cumulative_loss = 0.0
        cumulative_accuracy = 0.0

        self.__model.train()

        if self.__use_SAM:
            from models_methods.utility.bypass_bn import enable_running_stats, disable_running_stats

        for batch_idx, (inputs, targets) in enumerate(self.__data_loaders["train_loader"]):
            inputs, targets = inputs.to(self.__device), targets.to(self.__device)
            
            # first forward-backward step
            if self.__use_SAM:        
                enable_running_stats(self.__model) # disable batch norm running stats

            outputs = self.__model(inputs)

            loss = self.__loss_fn(outputs, targets)
            loss.mean().backward()
            
            if self.__use_SAM:
                self.__optimizer.first_step(zero_grad=True)
                # second forward-backward step
                disable_running_stats(self.__model)
                loss = self.__loss_fn(self.__model(inputs), targets)
                loss.mean().backward()
                self.__optimizer.second_step(zero_grad=True)
            else:
                self.__optimizer.step()
                self.__optimizer.zero_grad()
            
            samples += inputs.shape[0]
            cumulative_loss += loss.mean().item()
            _, predicted = outputs.max(dim=1)

            cumulative_accuracy += predicted.eq(targets).sum().item()

            if verbose and batch_idx % log_interval == 0:
                current_loss = cumulative_loss / samples
                current_accuracy = cumulative_accuracy / samples * 100
                print(f'Batch {batch_idx}/{len(self.__data_loaders["train_loader"])}, Loss: {current_loss:.4f}, Accuracy: {current_accuracy:.2f}%', end='\r')

        return cumulative_loss / samples, cumulative_accuracy / samples * 100

    def __test_step(self, test=False, eval=False, train=False):
        
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
        cumulative_accuracy = 0.

        self.__model.eval()

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.__device)
                targets = targets.to(self.__device)

                outputs = self.__model(inputs)

                loss = self.__loss_fn(outputs, targets)

                samples += inputs.shape[0]
                cumulative_loss += loss.mean().item() 
                _, predicted = outputs.max(1)

                cumulative_accuracy += predicted.eq(targets).sum().item()

        return cumulative_loss / samples, cumulative_accuracy / samples * 100

    def main(self,
             epochs=10,
             verbose_steps=True, # print after log_interval-learning steps
             log_interval=10): 

        from models_methods.utility.initialize import initialize
        initialize(self.__seed)
            
        self.__model.to(self.__device)
                
        # Log to TensorBoard
        if self.__trained == False:
            self.__trained = True
            print("Before training:")
            train_loss, train_accuracy = self.__test_step(train=True)
            val_loss, val_accuracy = self.__test_step(eval=True)
            test_loss, test_accuracy = self.__test_step(test=True)
            self.__log_values(self.__writer, self.__epoch, train_loss, train_accuracy, "Train")
            self.__log_values(self.__writer, self.__epoch, val_loss, val_accuracy, "Validation")
            self.__log_values(self.__writer, self.__epoch, test_loss, test_accuracy, "Test")
            self.__print_statistics(train_loss, train_accuracy, val_loss, val_accuracy, test_loss, test_accuracy)
        
        pbar = tqdm(range(epochs), desc="Training")
        for _ in pbar:
            train_loss, train_accuracy = self.__train_step(verbose=verbose_steps, log_interval=log_interval)
            # if scheduler:
            #     scheduler.step()
            val_loss, val_accuracy = self.__test_step(eval=True) 
            
            print("-----------------------------------------------------")
            self.__epoch += 1
            self.__log_values(self.__writer, self.__epoch, train_loss, train_accuracy, "Train")
            self.__log_values(self.__writer, self.__epoch, val_loss, val_accuracy, "Validation")

            pbar.set_postfix(train_loss=train_loss, train_accuracy=train_accuracy, val_loss=val_loss, val_accuracy=val_accuracy)

            if self.__use_early_stopping:
                self.__early_stopping(val_loss, self.__model)
                if self.__early_stopping.early_stop:
                    print("Early stopping")
                    break
        
        # Compute final evaluation results
        print("After training:")
        train_loss, train_accuracy = self.__test_step(train=True)
        val_loss, val_accuracy = self.__test_step(eval=True)
        test_loss, test_accuracy = self.__test_step(test=True)

        self.__log_values(self.__writer, self.__epoch, train_loss, train_accuracy, "Train")
        self.__log_values(self.__writer, self.__epoch, val_loss, val_accuracy, "Validation")
        self.__log_values(self.__writer, self.__epoch, test_loss, test_accuracy, "Test")

        self.__print_statistics(train_loss, train_accuracy, val_loss, val_accuracy, test_loss, test_accuracy)

        # Flush the logs to disk 
        self.__writer.flush()

        if self.__use_early_stopping == False:
            # save the model if early stopping was not used
            loss = self.get_statistics()["val_loss"]
            if val_loss < loss:
                torch.save(self.__model.state_dict(), 
                           self.__exp_name) 
            self.__update_statistics(train_loss, train_accuracy, val_loss, val_accuracy, test_loss, test_accuracy)

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

    def get_statistics(self):
        """
        Returns the last computed statistics
        """
        return {"train_loss": self.__train_loss,
                "train_accuracy": self.__train_accuracy,
                "val_loss": self.__val_loss,
                "val_accuracy": self.__val_accuracy,
                "test_loss": self.__test_loss,
                "test_accuracy": self.__test_accuracy}

    def __update_statistics(self, train_loss, train_accuracy, val_loss, val_accuracy, test_loss, test_accuracy):
        self.__train_loss = train_loss
        self.__train_accuracy = train_accuracy
        self.__val_loss = val_loss
        self.__val_accuracy = val_accuracy
        self.__test_loss = test_loss
        self.__test_accuracy = test_accuracy
    
    def __print_statistics(self, train_loss, train_accuracy, val_loss, val_accuracy, test_loss, test_accuracy):
        print(f"\tTraining loss {train_loss:.5f}, Training accuracy {train_accuracy:.2f}")
        print(f"\tValidation loss {val_loss:.5f}, Validation accuracy {val_accuracy:.2f}")
        print(f"\tTest loss {test_loss:.5f}, Test accuracy {test_accuracy:.2f}")
        print("-----------------------------------------------------")

    # tensorboard logging utilities
    def __log_values(self, writer, step, loss, accuracy, prefix):
        writer.add_scalar(f"{prefix}/loss", loss, step)
        writer.add_scalar(f"{prefix}/accuracy", accuracy, step)
        
    def __save_config(self, dataset_name, rho_SAM, momentum, weight_decay, lr, patience, delta):
        config = {
            'exp_path': self.__exp_name,
            'data': { 
                'batch_size':self.__data_loaders["train_loader"].batch_size,
                'dataset_name': dataset_name
            }, 
            'model': str(self.__model), 
            'optimizer': {
                'optimizer': str(self.__optimizer),
                'momentum': momentum,
                'weight_decay': weight_decay,
                'lr': lr,
                'use_SAM': self.__use_SAM,
                "rho_SAM": rho_SAM,
                'scheduler': str(self.__scheduler)
            } ,
            'loss_fn': {
                'loss_fn': str(self.__loss_fn),
                'smoothing': self.__loss_fn.label_smoothing
            },
            'device': str(self.__device),
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