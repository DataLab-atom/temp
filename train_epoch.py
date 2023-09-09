import torch
from utils import num_graphs
import numpy as np
from min_norm_solvers import MinNormSolver
import torch.nn.functional as F 
import torch.nn as nn
from torch.autograd import Variable
import time

def static_weight_loss(model, optimizer, loader, device,lastc, args):
    
    model.train()
    total_loss = 0
    total_loss_c = 0
    total_loss_o = 0
    total_loss_co = 0
    correct_o = 0

    for it, data in enumerate(loader):
    
        optimizer.zero_grad()
        data = data.to(device)    
        one_hot_target = data.y.view(-1)

        c_logs,o_logs,co_logs = model(data)

        uniform_target = torch.ones_like(c_logs, dtype=torch.float).to(device) / model.num_classes

        c_loss = F.kl_div(c_logs, uniform_target, reduction='batchmean')
        o_loss = F.nll_loss(o_logs, one_hot_target) 
        co_loss = F.nll_loss(co_logs, one_hot_target) 
        
        loss =  args.co*co_loss + args.o*o_loss + args.c*c_loss 
        start = time.time()
        loss.backward()
        mytime = (time.time() - start)*1000
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
        
        pred_o = o_logs.max(1)[1]
        correct_o += pred_o.eq(data.y.view(-1)).sum().item()
        total_loss += loss.item() * num_graphs(data)
        total_loss_c += c_loss.item() * num_graphs(data)
        total_loss_o += o_loss.item() * num_graphs(data)
        total_loss_co += co_loss.item() * num_graphs(data)
        optimizer.step()
    
    num = len(loader.dataset)
    total_loss = total_loss / num
    total_loss_c = total_loss_c / num
    total_loss_o = total_loss_o / num
    total_loss_co = total_loss_co / num
    correct_o = correct_o / num
    return mytime,lastc,total_loss, total_loss_c, total_loss_o, total_loss_co, correct_o,0

def get_parameters_grad(model):
    grads = []
    for param in model.Gnn_encoder.parameters():
        if param.grad is not None:
            grads.append(Variable(param.grad.data.clone(), requires_grad=False))

    return grads

def mgda_loss(model, optimizer, loader, device, lastc , args):
    model.train()
    total_loss = 0
    total_loss_c = 0
    total_loss_o = 0
    total_loss_co = 0
    correct_o = 0
    eval_random=args.with_random
    cration =  nn.SmoothL1Loss()
    mgda = []
    for it, data in enumerate(loader):
    
        optimizer.zero_grad()
        data = data.to(device)    
        one_hot_target = data.y.view(-1)

        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        
        #+++++++++++++++++++++++++++forward+++++++++++++++++++++++++++
        big_ = model.Gnn_encoder(x,edge_index,model.use_bns_conv)
        
        xo,_,_ = model.Acausation_sub_encoder(big_,edge_index,batch,eval_random)
        xc,_,_ = model.Causal_sub_encoder(big_,edge_index,batch,eval_random)
        
        c_logs = model.Acausation_Classifier(xc)        
        o_logs = model.Causal_Classifier(xo)
        co_logs = model.Causal_Interventions_Classifier(xc,xo,eval_random)

        uniform_target = torch.ones_like(c_logs, dtype=torch.float).to(device) / model.num_classes
        
        #+++++++++++++++++++++++++++++++++mgda++++++++++++++++++++++

        loss_data = {}
        grads = {}
        start = time.time()
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if len(lastc) - 1 < it:
            c_loss = F.kl_div(c_logs, uniform_target, reduction='batchmean')
            lastc.append(c_logs.clone().detach())    
        else:
            c_loss = cration(c_logs,lastc[it])
            lastc[it] = c_logs.clone().detach()
        
        loss_data['c'] = c_loss.data
        c_loss.backward(retain_graph=True)
        grads['c'] = get_parameters_grad(model)
        model.zero_grad()

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  

        o_loss = F.nll_loss(o_logs, one_hot_target)
        loss_data['o'] = o_loss.data
        o_loss.backward(retain_graph=True)
        grads['o'] =  get_parameters_grad(model)
        model.zero_grad()

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
        
        co_loss = F.nll_loss(co_logs, one_hot_target) 
        
        if args.mgda_with_double_co:
            co_logs_2 = model.Causal_Interventions_Classifier(xc,xo,eval_random)
            co_loss_2 = torch.pairwise_distance(co_logs,co_logs_2) 
            co_loss + co_loss + co_loss_2
        
        loss_data['co'] = co_loss.data
        co_loss.backward(retain_graph=True)
        grads['co'] =  get_parameters_grad(model)
        model.zero_grad()
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
        
        loss_name = ['o','c']
        gn = MinNormSolver.gradient_normalizers(grads, loss_data, args.mgda_model)
        for name in loss_name:
            if gn[name] < 1e-3:
                gn[name] = torch.tensor(1e-3)

        for t in loss_data:
            for gr_i in range(len(grads[t])):
                grads[t][gr_i] = grads[t][gr_i] / gn[t].to(grads[t][gr_i].device)
        sol, _ = MinNormSolver.find_min_norm_element_FW([grads[t] for t in loss_name])
        sol = {k:sol[i] for i, k in enumerate(loss_name)}
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
         
        loss = sol['c']*c_loss +  sol['o']*o_loss + co_loss      
        loss.backward()
        mytime = (time.time() - start)*1000
        optimizer.step()
        mgda.append([float(sol['c']) , float(sol['o'])])  
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 

        pred_o = o_logs.max(1)[1]
        correct_o += pred_o.eq(data.y.view(-1)).sum().item()
        total_loss += loss.item() * num_graphs(data)
        total_loss_c += c_loss.item() * num_graphs(data)
        total_loss_o += o_loss.item() * num_graphs(data)
        total_loss_co += co_loss.item() * num_graphs(data)
        
    mgda = torch.tensor(mgda)
    num = len(loader.dataset)
    total_loss = total_loss / num
    total_loss_c = total_loss_c / num
    total_loss_o = total_loss_o / num
    total_loss_co = total_loss_co / num
    correct_o = correct_o / num
    return mytime,lastc,total_loss, total_loss_c, total_loss_o, total_loss_co, correct_o,mgda


# handle for flexible external changes to the training mode
funcs = {
    'swl':static_weight_loss,
    'mgda':mgda_loss,
}

