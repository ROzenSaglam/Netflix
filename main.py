import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

# TODO: Your code here
K = np.array([1,2,3,4])
seeds = np.array([0,1,2,3,4])

def lowest_cost(K, X, seeds):
    
    Ct = []
    for k in K:
    
        mixture_list =  []
        post_list = []
        for seed in seeds:
            mixture, post= common.init(X, k, seed)
            mixture_list.append(mixture)
            post_list.append(post)
        
        cost_list = []
        for mixture,post in zip(mixture_list, post_list):
            
            mixture, post, cost = kmeans.run(X, mixture, post)
            cost_list.append(cost)
        cost_min = np.min(cost_list)
        Ct.append(cost_min)
    return Ct

lowest_cost(K, X, seeds)
            