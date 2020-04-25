# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Market_Basket_Optimisation.csv',header=None)

#basket=[['A','C','D','F','G'],['A','B','C','D','F'],['C','D','E'],['A','D','F'],['A','C','D','E','F'],['B','C','D','E','F','G']]

# apriori takes a list of list as the input
transaction=[]
for i in range(0, dataset.shape[0]):
    transaction.append([str(dataset.values[i,j]) for j in range(0, dataset.shape[1])])
    
#training the apriori model
from apyori import apriori
rules = apriori(transaction, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

#rules = apriori(basket, min_support = 0.5, min_confidence = 0.75, min_lift = 0, min_length = 2)

# Visualising the results
results = list(rules)