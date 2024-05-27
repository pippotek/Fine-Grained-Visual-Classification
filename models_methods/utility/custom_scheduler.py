class StepLR_SAM:
    """
    Needed for SAM optimizer because it doesn't work with torch schedulers.
    """
    def __init__(self, optimizer, step_size=1, gamma=0.01):
        self.optimizer = optimizer
        self.gamma = gamma
        self.step_size = step_size
        self.base_lrs = optimizer.param_groups[0]["lr"]
        self.last_lr = optimizer.param_groups[0]["lr"]
        self.epoch = 0

    def step(self, verbose=False):

        self.epoch += 1
        if self.epoch % self.step_size == 0:
            self.last_lr = self.optimizer.param_groups[0]["lr"]
            self.optimizer.param_groups[0]["lr"] *= self.gamma 
            if verbose:
                print(f"New lr: {self.optimizer.param_groups[0]['lr']}")
    
    def state_dict(self):
        return {"step_size":self.step_size,
                "gamma": self.gamma,
                "base_lrs": self.base_lrs,
                "epoch": self.epoch,
                "last_lr": self.last_lr}
    
    def load_state_dict(self, state_dict):
        self.step_size = state_dict["step_size"]
        self.gamma = state_dict["gamma"]
        self.base_lrs = state_dict["base_lrs"]
        self.epoch = state_dict["epoch"]
        self.last_lr = state_dict["last_lr"]