import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        """
        rho: radius around the current parameter values where we look for the local maximum
        """
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        Compute e_w and add it to the parameters. Since we'll need to current parameters to perform 
        the second step, we store them in the state dictionary. 
        """
        grad_norm = self._grad_norm() 
        for group in self.param_groups: 
            scale = group["rho"] / (grad_norm + 1e-12) # scaling factor for gradients

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone() 
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        Update the weights by evaluating the graidents at the local maximum. We first restore the current parameters 
        and then perform the actual update.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        """
        Calculates the norm of the gradients for all parameters in the model
        """
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm( # calculate L2 Norm of the norm of the gradients
                    torch.stack([ # stack all the parameter gradients together
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device) # L2 norm of the scaled gradient
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
