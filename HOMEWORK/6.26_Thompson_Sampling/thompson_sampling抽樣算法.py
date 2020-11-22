import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os  #工作路徑用
import random
import math

os.chdir("D:\Machine Learning A-Z Chinese Template Folder\Part 6 - Reinforcement Learning\Section 26 - Thompson Sampling");

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Implementing Tthompson samplin
N = 10000
ad_count = 10
numbers_of_rewards_1 = [0]* ad_count
numbers_of_rewards_0 = [0]* ad_count
random_number_record = [0]* ad_count
random_number_record_totle = []
ads_selected = []
totle_reward = 0

for n in range(0, N): 
    ad_number = 0
    max_random = 0
    
    for i in range(0,ad_count):
       random_beta = random.betavariate(numbers_of_rewards_1[i] +1 , numbers_of_rewards_0[i] +1)
       random_number_record[i] = random_beta
       #random_number_record = ['{:.4f}'.format(i) for i in random_number_record] 
       if random_beta > max_random:
           max_random = random_beta
           ad_number = i
            
    ads_selected.append(ad_number)
    random_number_record_totle.append(random_number_record)
    reward = dataset.values[n, ad_number]
    if reward == 1:
        numbers_of_rewards_1[ad_number] =  numbers_of_rewards_1[ad_number] + 1
    else:
        numbers_of_rewards_0[ad_number] =  numbers_of_rewards_0[ad_number] + 1
    
    totle_reward += reward
    

plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('ADs')
plt.show
