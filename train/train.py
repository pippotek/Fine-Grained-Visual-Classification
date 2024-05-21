import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import os
os.chdir("/home/peppe/01_Study/01_University/Semester/2/Intro_to_ML/Project/Code") # to import modules from other directories
print("Warning: the working directory was changed to", os.getcwd())

class Trainer:

    def __init__(self, data_loaders: dict, model, optimizer, loss_fn, device, SAM=False, smoothing=0.1):
        self.data_loaders = data_loaders
        self.model = model 
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.SAM = SAM
        self.smoothing = smoothing

    def train_step(model, data_loader, optimizer, loss_fn, device, SAM=False, smoothing=0.1, verbose=False, log_interval=10):
        samples = 0.0
        cumulative_loss = 0.0
        cumulative_accuracy = 0.0

        model.train()

        if SAM:
            from models_methods.utility.bypass_bn import enable_running_stats, disable_running_stats

        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # first forward-backward step
            if SAM:        
                enable_running_stats(model) # disable batch norm running stats

            outputs = model(inputs)

            if SAM:
                loss = loss_fn(outputs, targets, smoothing=smoothing)
            else:
                loss = loss_fn(outputs, targets)

            loss.mean().backward()
            
            if SAM:
                optimizer.first_step(zero_grad=True)
                # second forward-backward step
                disable_running_stats(model)
                loss = loss_fn(model(inputs), targets, smoothing=smoothing)
                loss.mean().backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.step()
                optimizer.zero_grad()
            
            samples += inputs.shape[0]
            cumulative_loss += loss.mean().item()
            _, predicted = outputs.max(dim=1)

            cumulative_accuracy += predicted.eq(targets).sum().item()

            if verbose and batch_idx % log_interval == 0:
                current_loss = cumulative_loss / samples
                current_accuracy = cumulative_accuracy / samples * 100
                print(f'Batch {batch_idx}/{len(data_loader)}, Loss: {current_loss:.4f}, Accuracy: {current_accuracy:.2f}%', end='\r')

        return cumulative_loss / samples, cumulative_accuracy / samples * 100

    def test_step(model, data_loader, loss_fn, device):
        samples = 0.
        cumulative_loss = 0.
        cumulative_accuracy = 0.

        model.eval()

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)

                loss = loss_fn(outputs, targets)

                samples += inputs.shape[0]
                cumulative_loss += loss.mean().item() 
                _, predicted = outputs.max(1)

                cumulative_accuracy += predicted.eq(targets).sum().item()

        return cumulative_loss / samples, cumulative_accuracy / samples * 100

    # tensorboard logging utilities
    def __log_values(writer, step, loss, accuracy, prefix):
        writer.add_scalar(f"{prefix}/loss", loss, step)
        writer.add_scalar(f"{prefix}/accuracy", accuracy, step)

    def main(model,
         optimizer,
         loss_fn,
         data_loaders: dict,
         train_step: callable,
         test_step: callable,
         device,
         epochs=10,
         exp_name=None,
         exp_path="/home/peppe/01_Study/01_University/Semester/2/Intro_to_ML/Project/Code/experiments/",
         use_early_stopping=True,
         patience=5,
         delta=1e-3,
         scheduler=None,
         verbose_steps=True, # print after log_interval-learning steps
         log_interval=10,
         use_SAM=False, # if SAM=True then loss_fn must be smooth_cross_entropy with smoothing >= 0.07
         smoothing=0.1,
         seed=None): 
    
        assert os.path.exists(f"{exp_path}"), "Experiment path does not exist"
        assert seed is not None, "Seed must be specified"

        from models_methods.utility.initialize import initialize
        initialize(seed=42)

        if use_SAM == True: 
            
            from models_methods.utility.smooth_cross_entropy import smooth_crossentropy
            from models_methods.methods.sam import SAM 

            assert smoothing >= 0.07, "smoothing must be >= 0.7 when using SAM"
            assert loss_fn == smooth_crossentropy, "loss function must be smooth_crossentropy when using SAM"   
            optimizer = SAM(model.parameters(), 
                            optimizer, 
                            rho=2, 
                            adaptive=True, 
                            lr=0.1, momentum=0.9, weight_decay=0.0005)
                
        # Create a logger for the experiment
        writer = SummaryWriter(log_dir=f"{exp_path + exp_name}")

        if use_early_stopping:
            from models_methods.utility.early_stopping import EarlyStopping
            early_stopping = EarlyStopping(patience=patience, 
                                        delta=delta,
                                        path=f"{exp_path + exp_name + '/checkpoint.pt'}",)
            
        model.to(device)
        
        # Computes evaluation results before training
        print("Before training:")
        train_loss, train_accuracy = test_step(model, data_loaders["train_loader"], loss_fn,device=device)
        val_loss, val_accuracy = test_step(model, data_loaders["val_loader"], loss_fn,device=device)
        test_loss, test_accuracy = test_step(model, data_loaders["test_loader"], loss_fn,device=device)
        
        # Log to TensorBoard
        self.__log_values(writer, -1, train_loss, train_accuracy, "Train")
        self.__log_values(writer, -1, val_loss, val_accuracy, "Validation")
        self.__log_values(writer, -1, test_loss, test_accuracy, "Test")

        print(f"\tTraining loss {train_loss:.5f}, Training accuracy {train_accuracy:.2f}")
        print(f"\tValidation loss {val_loss:.5f}, Validation accuracy {val_accuracy:.2f}")
        print(f"\tTest loss {test_loss:.5f}, Test accuracy {test_accuracy:.2f}")
        print("-----------------------------------------------------")
        
        pbar = tqdm(range(epochs), desc="Training")
        for e in pbar:
            train_loss, train_accuracy = train_step(model, data_loaders["train_loader"], optimizer, loss_fn, 
                                                    device=device, SAM=use_SAM, verbose=verbose_steps, log_interval=log_interval)
            #if scheduler:
            #    scheduler.step()
            val_loss, val_accuracy = test_step(model, data_loaders["val_loader"], loss_fn,device=device)
            
            print("-----------------------------------------------------")
            
            # Logs to TensorBoard
            self.__log_values(writer, e, train_loss, train_accuracy, "Train")
            self.__log_values(writer, e, val_loss, val_accuracy, "Validation")

            pbar.set_postfix(train_loss=train_loss, train_accuracy=train_accuracy, val_loss=val_loss, val_accuracy=val_accuracy)

            if use_early_stopping:
                early_stopping(val_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
        # Compute final evaluation results
        print("After training:")
        train_loss, train_accuracy = test_step(model, data_loaders["train_loader"], loss_fn,device=device)
        val_loss, val_accuracy = test_step(model, data_loaders["val_loader"], loss_fn,device=device)
        test_loss, test_accuracy = test_step(model, data_loaders["test_loader"], loss_fn,device=device)

        # Log to TensorBoard
        self.__log_values(writer, epochs + 1, train_loss, train_accuracy, "Train")
        self.__log_values(writer, epochs + 1, val_loss, val_accuracy, "Validation")
        self.__log_values(writer, epochs + 1, test_loss, test_accuracy, "Test")

        print(f"\tTraining loss {train_loss:.5f}, Training accuracy {train_accuracy:.2f}")
        print(f"\tValidation loss {val_loss:.5f}, Validation accuracy {val_accuracy:.2f}")
        print(f"\tTest loss {test_loss:.5f}, Test accuracy {test_accuracy:.2f}")
        print("-----------------------------------------------------")

        # Closes the logger
        writer.close()

        # Let's return the net
        return model