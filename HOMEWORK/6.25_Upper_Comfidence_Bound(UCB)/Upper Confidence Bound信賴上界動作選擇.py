import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os  #工作路徑用
import math

os.chdir("D:\Machine Learning A-Z Chinese Template Folder\Part 6 - Reinforcement Learning\Section 25 - Upper Confidence Bound (UCB)");

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Implementing UCB
N = 10000
ad_count = 10
numbers_of_selections = [0] * ad_count
sums_of_rewards = [0] * ad_count
ads_selected = []
totle_reward = 0

for n in range(0, N): 
    ad_number = 0
    max_upper_bound = 0
    
    for i in range(0,ad_count):
        if (numbers_of_selections[i] > 0):
            average_reward = sums_of_rewards[i]/numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n+1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad_number = i
            
    ads_selected.append(ad_number)
    reward = dataset.values[n, ad_number]
    numbers_of_selections[ad_number] += 1
    sums_of_rewards[ad_number] += reward
    totle_reward += reward
    

plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('ADs')
plt.show
    
    
    