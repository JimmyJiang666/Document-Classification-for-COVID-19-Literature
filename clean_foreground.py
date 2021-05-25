import pandas as pd
import numpy as np
import random

train_set = pd.read_csv('train.csv')

abstract = [str(x) for x in train_set['abstract']]

import os
cwd = os.getcwd()

# N = 5000
import os.path

Y_train = [str(x).split(';') for x in train_set['label']]

labels = ['Diagnosis', 'Transmission', 'Case Report', 'General Info', 'Treatment', 'Mechanism', 'Epidemic Forecasting', 'Prevention', 'nan']

# label_item = labels[2]
# label_item = labels[3]
label_item = labels[6]

# print(abstract[29445])
# print(Y_train[29445])


# print(abstract[20808])
# print(Y_train[20808])


random_abstract = []

for i in range(len(abstract)):
	if label_item in Y_train[i]:
		random_abstract.append(abstract[i])
N = len(random_abstract)


# random subset foreground
# for i in range(N):
# 	print(i)
# 	index = random.randint(0,len(abstract)-1)
# 	while label_item not in Y_train[index] or abstract[index]=="nan": #skip those are not labeled what we want and skip those without abstracts
# 		index = random.randint(0,len(abstract)-1)
# 	if label_item in Y_train[index]:# this if statement should always be true after the former while loop
# 		print(label_item, "->", Y_train[index]," inDex: ", index)
# 		ele = abstract.pop(index)
# 		Y_train.pop(index)
# 		random_abstract.append(ele)



# seen_inDex = []
# while len(random_abstract) < N:
# 	inDex = random.randint(0,len(abstract)-1)
# 	if inDex in seen_inDex:
# 		continue
# 	else:
# 		seen_inDex.append(inDex)
# 	if label_item in Y_train[inDex] and abstract[inDex]!= "nan":#skip those are not labeled what we want and skip those without abstracts
# 		# print(label_item, "->", Y_train[inDex]," inDex: ", inDex)
# 		random_abstract.append(abstract[inDex])



print('length of ' + label_item + ' random_abstract: ', len(random_abstract) )

# print(random_abstract[1])

# print(abstract[1])
# print()
# print(random_abstract[1])

label_item = label_item.replace(' ', '_')
subdirectory = str(N)+label_item
try:
    os.mkdir(subdirectory)
except Exception:
    pass

list_file = open(str(N)+'_'+label_item+'.list','w')

# print(random_abstract[0])

for idx,a in enumerate(random_abstract):
	with open(os.path.join(subdirectory,label_item+"_"+str(idx)+".txt"), "w") as f:
		list_file.write(subdirectory+'/'+label_item+"_"+str(idx)+".txt"+"\n")
		f.write(a)
		f.close()


# for idx,a in enumerate(abstract):
# 	if idx < N:
# 		with open(os.path.join(subdirectory,"foreground_"+str(idx)+".txt"), "a") as f:
# 			f.write(a)
# 			f.close()

