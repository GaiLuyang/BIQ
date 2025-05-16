import torch
import torch.distributions as distributions
import copy

def Quantization(param_client_data, param_client_model_k_data, Ran, task): 
    
    Delta_data = param_client_data.to(task.device) - param_client_model_k_data.to(task.device)
    original_shape = Delta_data.shape
    Delta_data_1d = Delta_data.view(-1)
    Delta_data_1d = Delta_data_1d.clamp_(-Ran, Ran) 

    
    if task.opt == 'FedBIQ':
        Delta_data_1d = BIQ(Delta_data_1d, Ran, task)
    elif task.opt == 'FedWBIQ':
        Delta_data_1d = WBIQ(Delta_data_1d, Ran, task)
    
    Delta_data_new = Delta_data_1d.view(original_shape).to(task.device)
    Q_param = param_client_model_k_data + Delta_data_new
    return Q_param

def best_Q(data, Ran, Q_step):
    h = torch.floor((data + Ran)/Q_step).to(torch.int64)
    return h

def BIQ(data, Ran, task): 
    Q_ruler_l = -torch.tensor(Ran).repeat(len(data)).to(task.device)
    Q_ruler_r = torch.tensor(Ran).repeat(len(data)).to(task.device)
    for i in range(0, task.b):
        mask = (Q_ruler_l + Q_ruler_r)/2 <= data 
        Q_ruler_l = torch.where(~mask, Q_ruler_l, (Q_ruler_l + Q_ruler_r)/2)
        Q_ruler_r = torch.where(mask, Q_ruler_r, (Q_ruler_l + Q_ruler_r)/2)
    return (Q_ruler_l + Q_ruler_r)/2

def WBIQ(data, Ran, task): 
    weight = torch.zeros(len(data)).to(task.device) 
    sum_weight = (torch.zeros(len(data)) + task.b).to(task.device) 
    Q_ruler_l = -torch.tensor(Ran).repeat(len(data)).to(task.device)
    Q_ruler_r = torch.tensor(Ran).repeat(len(data)).to(task.device)
    for i in range(0, task.b):
        mask = (Q_ruler_l + Q_ruler_r)/2 <= data 
        weight[mask] += 1 
        Q_ruler_l = torch.where(~mask, Q_ruler_l, (Q_ruler_l + Q_ruler_r)/2)
        Q_ruler_r = torch.where(mask, Q_ruler_r, (Q_ruler_l + Q_ruler_r)/2)
    return (1-weight/sum_weight)*Q_ruler_l + (weight/sum_weight)*Q_ruler_r