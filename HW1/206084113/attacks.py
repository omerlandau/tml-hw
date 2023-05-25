import torch
import torch.nn as nn
import torch.nn.functional as F

class PGDAttack:
    """
    White-box L_inf PGD attack using the cross-entropy loss
    """
    def __init__(self, model, eps=8/255., n=50, alpha=1/255.,
                 rand_init=True, early_stop=True):
        """
        Parameters:
        - model: model to attack
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: step size at each iteration
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a 
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.model = model
        self.n = n
        self.alpha = alpha
        self.eps = eps
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def execute(self, x, y, targeted=False):
        """
        Executes the attack on a batch of samples x. y contains the true labels 
        in case of untargeted attacks, and the target labels in case of targeted 
        attacks. The method returns the adversarially perturbed samples, which
        lie in the ranges [0, 1] and [x-eps, x+eps]. The attack optionally 
        performs random initialization and early stopping, depending on the 
        self.rand_init and self.early_stop flags.
        """
        adv_samp = x.clone().detach().requires_grad_(True).to(x.device)
        
        if self.rand_init:
            adv_samp = adv_samp + \
                torch.empty_like(adv_samp).uniform_(-self.eps, self.eps)
            adv_samp = torch.clamp(adv_samp, min=0, max=1).detach()
        
        for _ in range(self.n):
            _adv_samp = adv_samp.clone().detach().requires_grad_(True)
            out_img = self.model(_adv_samp)
            if self.early_stop:
                pred = out_img.argmax(dim=1)
                if targeted and (pred == y).all():
                    break
                elif not targeted and (pred != y).all(): 
                    break

            loss = self.loss_func(out_img, y)
            loss = loss.sum()
            loss.backward()

            with torch.no_grad():
                grad = _adv_samp.grad.sign() * self.alpha
                if targeted:
                    adv_samp -= grad
                else:
                    adv_samp += grad

            delta = torch.clamp(adv_samp - x,
                                min=-self.eps, max=self.eps)
            adv_samp = torch.clamp(x + delta, min=0, max=1).detach()

        assert (adv_samp - x).abs().max() <= self.eps + 1e-4

        return adv_samp
      
      
class NESBBoxPGDAttack:
    """
    Query-based black-box L_inf PGD attack using the cross-entropy loss, 
    where gradients are estimated using Natural Evolutionary Strategies 
    (NES).
    """
    def __init__(self, model, eps=8/255., n=50, alpha=1/255., momentum=0.,
                 k=200, sigma=1/255., rand_init=True, early_stop=True):
        """
        Parameters:
        - model: model to attack
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: PGD's step size at each iteration
        - momentum: a value in [0., 1.) controlling the "weight" of
             historical gradients estimating gradients at each iteration
        - k: the model is queries 2*k times at each iteration via 
              antithetic sampling to approximate the gradients
        - sigma: the std of the Gaussian noise used for querying
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a 
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.model = model
        self.eps = eps
        self.n = n
        self.alpha = alpha
        self.momentum = momentum
        self.k = k
        self.sigma=sigma
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def execute(self, x, y, targeted=False):
        """
        Executes the attack on a batch of samples x. y contains the true labels 
        in case of untargeted attacks, and the target labels in case of targeted 
        attacks. The method returns:
        1- The adversarially perturbed samples, which lie in the ranges [0, 1] 
            and [x-eps, x+eps].
        2- A vector with dimensionality len(x) containing the number of queries for
            each sample in x.
        """
        adv_samp = x.clone().detach().requires_grad_(True).to(x.device)
        
        if self.rand_init:
            adv_samp = adv_samp + \
                torch.empty_like(adv_samp).uniform_(-self.eps, self.eps)
            adv_samp = torch.clamp(adv_samp, min=0, max=1).detach() 
        
        failed_samps = torch.ones(len(x), dtype=torch.bool)
        grad_calc = torch.zeros_like(x)
        queries = torch.zeros(len(x), dtype=torch.int32)
        
        for _ in range(self.n):
            grad = torch.zeros_like(x[failed_samps])
            for _ in range(self.k):
                # creating the gaussian noise 
                gaussian_noise = torch.randn_like(x[failed_samps])

                # estimating the around the noise 
                xp_n = torch.clamp(adv_samp[failed_samps] + gaussian_noise * self.sigma, min=0, max=1)
                xm_n = torch.clamp(adv_samp[failed_samps] - gaussian_noise * self.sigma, min=0, max=1)
                with torch.no_grad():
                        outputs_pos = self.model(xp_n)
                        outputs_neg = self.model(xm_n)
                        l_p = self.loss_func(outputs_pos, y[failed_samps]) 
                        l_m = self.loss_func(outputs_neg, y[failed_samps])
                loss = l_p - l_m
                
                grad += gaussian_noise * loss.unsqueeze(1).unsqueeze(2).unsqueeze(3) * self.alpha
                
            queries[failed_samps] += 2 * self.k 
            grad /= (2 * self.k * self.sigma)
            grad_calc[failed_samps] = self.momentum * grad_calc[failed_samps] + (1 - self.momentum) * grad
            
            with torch.no_grad():
                grad = torch.sign(grad_calc[failed_samps]) * self.alpha
                if targeted:
                    adv_samp[failed_samps] -= grad
                else:
                    adv_samp[failed_samps] += grad

            delta = torch.clamp(adv_samp - x,
                                min=-self.eps, max=self.eps)
            adv_samp = torch.clamp(x + delta, min=0, max=1).detach()

            out_img = self.model(adv_samp)
            pred = out_img.argmax(dim=1)
            if targeted:
                failed_samps = (pred != y)
                if self.early_stop and (pred == y).all():
                    break
            else:
                failed_samps = (pred == y)
                if self.early_stop and (pred != y).all(): 
                    break

        assert (adv_samp - x).abs().max() <= self.eps + 1e-4

        return adv_samp, queries


class PGDEnsembleAttack:
    """
    White-box L_inf PGD attack against an ensemble of models using the 
    cross-entropy loss
    """
    def __init__(self, models, eps=8/255., n=50, alpha=1/255.,
                 rand_init=True, early_stop=True):
        """
        Parameters:
        - models (a sequence): an ensemble of models to attack (i.e., the
              attack aims to decrease their expected loss)
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: PGD's step size at each iteration
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a 
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.models = models
        self.n = n
        self.alpha = alpha
        self.eps = eps
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss()

    def execute(self, x, y, targeted=False):
        """
        Executes the attack on a batch of samples x. y contains the true labels 
        in case of untargeted attacks, and the target labels in case of targeted 
        attacks. The method returns the adversarially perturbed samples, which
        lie in the ranges [0, 1] and [x-eps, x+eps].
        """
        adv_samp = x.clone().detach().requires_grad_(True).to(x.device)
        
        if self.rand_init:
            adv_samp = adv_samp + \
                torch.empty_like(adv_samp).uniform_(-self.eps, self.eps)
            adv_samp = torch.clamp(adv_samp, min=0, max=1).detach()
      
        for _ in range(self.n):
            _adv_samp = adv_samp.clone().detach().requires_grad_(True)
            out_img = self.models[0](adv_samp)
            if self.early_stop:
                pred = out_img.argmax(dim=1)
                if targeted and (pred == y).all():
                    break
                elif not targeted and (pred != y).all(): 
                    break
            
            grad = torch.zeros_like(_adv_samp)
            for model in self.models:
                outputs = model(_adv_samp)
                loss = self.loss_func(outputs, y)
                grad += torch.autograd.grad(loss.sum(), _adv_samp)[0].detach()
            
            if targeted:
                  adv_samp += - self.alpha * torch.sign(grad)
            else:
                  adv_samp += self.alpha * torch.sign(grad)
            
            delta = torch.clamp(adv_samp - x,
                                min=-self.eps, max=self.eps)
            adv_samp = torch.clamp(x + delta, min=0, max=1).detach()
            
        assert (adv_samp - x).abs().max() <= self.eps + 1e-4

        return adv_samp
