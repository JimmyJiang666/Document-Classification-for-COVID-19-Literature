import re
pattern = '[0-9]+\|a\|'

file1 = open('NCBItrainset_corpus.txt', 'r')
Lines1 = file1.readlines()

file2 = open('NCBItestset_corpus.txt', 'r')
Lines2 = file2.readlines()

file3 = open('NCBIdevelopset_corpus.txt', 'r')
Lines3 = file3.readlines()

Lines = Lines1 + Lines2 + Lines3
count = 0


N = 500
import os.path

subdirectory = str(N)+"background"
try:
    os.mkdir(subdirectory)
except Exception:
    pass

list_file = open('background.list','w')

for line in Lines:
    head = re.search(pattern,line)
    if head and count < N:
    	with open(os.path.join(subdirectory,"background_"+str(count)+".txt"), "a") as f:
    		list_file.write(subdirectory+'/'+"background_"+str(count)+".txt"+"\n")
    		f.write(line.lstrip(head.group()).rstrip())
    		f.close()
    	count += 1
# print(count)