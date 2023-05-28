import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from math import ceil

from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint

def free_adv_train(model, data_tr, criterion, optimizer, lr_scheduler, \
                   eps, device, m=4, epochs=100, batch_size=128, dl_nw=10):
    """
    Free adversarial training, per Shafahi et al.'s work.
    Arguments:
    - model: randomly initialized model
    - data_tr: training dataset
    - criterion: loss function (e.g., nn.CrossEntropyLoss())
    - optimizer: optimizer to be used (e.g., SGD)
    - lr_scheduer: scheduler for updating the learning rate
    - eps: l_inf epsilon to defend against
    - device: device used for training
    - m: # times a batch is repeated
    - epochs: "virtual" number of epochs (equivalent to the number of 
        epochs of standard training)
    - batch_size: training batch_size
    - dl_nw: number of workers in data loader
    Returns:
    - trained model
    """
    # init data loader
    loader_tr = DataLoader(data_tr,
                           batch_size=batch_size,
                           shuffle=True,
                           pin_memory=True,
                           num_workers=dl_nw)
                           

    # init delta (adv. perturbation)    

    tmp_input, _ = next(iter(loader_tr))
    
    delta = torch.zeros_like(tmp_input, device=device)

    # total number of updates

    totatl_updates_num = int(np.ceil(epochs/m))

    # when to update lr

    scheduler_step_iters = int(np.ceil(len(data_tr)/batch_size))

    sched_update = 0

    # train 

    for epoch in range(totatl_updates_num):
        for _, data in enumerate(loader_tr, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            for _ in range(m):
                noise = Variable(delta[:inputs.size(0)], requires_grad=True).to(device)
                noisy_input = inputs + noise

                optimizer.zero_grad()
                output = model(noisy_input)
                loss = criterion(output, labels)

                
                loss.backward()

                noise_update = eps*torch.sign(noise.grad)
                delta[:inputs.size(0)] += noise_update
                delta.clamp_(-eps, eps)

                optimizer.step()
                sched_update += 1

                if sched_update%scheduler_step_iters == 0:
                    lr_scheduler.step()
    
    # done
    return model


class SmoothedModel():
    """
    Use randomized smoothing to find L2 radius around sample x,
    s.t. the classification of x doesn't change within the L2 ball
    around x with probability >= 1-alpha.
    """

    ABSTAIN = -1

    def __init__(self, model, sigma):
        self.model = model
        self.sigma = sigma

    def _sample_under_noise(self, x, n, batch_size):
        """
        Classify input x under noise n times (with batch size 
        equal to batch_size) and return class counts (i.e., an
        array counting how many times each class was assigned the
        max confidence).
        """

        def count_err(array, size):
            counts = np.zeros(size, dtype=int)
            for i in array:
                counts[i] += 1
            return counts
        
        nc = 4 # There are four classes in the SimpleCNN model, 
               # if this parameter can change, please add it to the class.
        
        with torch.no_grad():
            counts = np.zeros(nc, dtype=int)
            for _ in range(ceil(n / batch_size)):
                current_bs = min(batch_size, n)
                n -= current_bs
                batch = x.repeat((current_bs, 1, 1, 1))
                noise = torch.randn_like(batch).to("cuda") * self.sigma
                predictions = self.model(batch + noise).argmax(1)
                counts += count_err(predictions.cpu().numpy(), nc)
            return counts
        
    def certify(self, x, n0, n, alpha, batch_size):
        """
        Arguments:
        - model: pytorch classification model (preferably, trained with
            Gaussian noise)
        - sigma: Gaussian noise's sigma, for randomized smoothing
        - x: (single) input sample to certify
        - n0: number of samples to find prediction
        - n: number of samples for radius certification
        - alpha: confidence level
        - batch_size: batch size to use for inference
        Outputs:
        - prediction / top class (ABSTAIN in case of abstaining)
        - certified radius (0. in case of abstaining)
        """
        
        # find prediction (top class c)

        c = self._sample_under_noise(x,n0,batch_size)
        c = c.argmax()

        c_estimate = self._sample_under_noise(x, n, batch_size)
        p_c = c_estimate[c].item()
        
        # compute lower bound on p_c
        l_b = proportion_confint(p_c, n, alpha=2*alpha, method="beta")[0]
        
        if l_b < 0.5:
            return SmoothedModel.ABSTAIN, 0.0
        
        radius = self.sigma* norm.ppf(l_b)
        return c, radius
        

class NeuralCleanse:
    """
    A method for detecting and reverse-engineering backdoors.
    """

    def __init__(self, model, dim=(1, 3, 32, 32), lambda_c=0.0005,
                 step_size=0.005, niters=2000):
        """
        Arguments:
        - model: model to test
        - dim: dimensionality of inputs, masks, and triggers
        - lambda_c: constant for balancing Neural Cleanse's objectives
            (l_class + lambda_c*mask_norm)
        - step_size: step size for SGD
        - niters: number of SGD iterations to find the mask and trigger
        """
        self.model = model
        self.dim = dim
        self.lambda_c = lambda_c
        self.niters = niters
        self.step_size = step_size
        self.loss_func = nn.CrossEntropyLoss()

    def find_candidate_backdoor(self, c_t, data_loader, device):
        """
        A method for finding a (potential) backdoor targeting class c_t.
        Arguments:
        - c_t: target class
        - data_loader: DataLoader for test data
        - device: device to run computation
        Outputs:
        - mask: 
        - trigger: 
        """
        # randomly initialize mask and trigger in [0,1]

        mask = torch.rand((self.dim[2],self.dim[3]), requires_grad=True, device=device)
        trigger = torch.rand(self.dim, requires_grad=True, device=device)

        # run self.niters of SGD to find (potential) trigger and mask

        sgd_count = 0

        while sgd_count <= self.niters:
            for _, data in enumerate(data_loader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                
                t_i = (1 - mask)*inputs + mask*trigger

                outputs = self.model(t_i)

                # build a misclassified labels for the batch all are c_t

                c_t = int(c_t)
                m_labels = torch.ones(labels.size(), dtype=int).to(device)
                m_labels*=c_t

                loss = self.loss_func(outputs, m_labels)
                # L1 norm regulator
                loss += self.lambda_c*mask.abs().sum().to(device)
                loss.backward()

                with torch.no_grad():
                    mask -= mask.grad.sign() * self.step_size
                    trigger -= trigger.grad.sign() * self.step_size

                    torch.clamp_(mask, 0, 1)
                    torch.clamp_(trigger, 0, 1)

                    mask.grad.zero_()
                    trigger.grad.zero_()

                    sgd_count += 1

        # done
        return mask.repeat((1,3,1,1)), trigger
