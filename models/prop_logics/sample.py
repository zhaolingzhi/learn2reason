#coding=utf-8

leaves = ['0', '1', 'and', 'or', 'not', '(', ')']
logics = ['F', 'T', '^', 'v', '~', '(', ')']


file="sample.txt"

line=[]

for i in range(len(logics)):
    for j in range(len(logics)):
        line.append(logics[i]+logics[j]+"\n")

with open(file,'w') as f:
    f.writelines(line)
