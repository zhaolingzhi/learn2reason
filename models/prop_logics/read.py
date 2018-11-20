import cPickle as pickle

with open("good_params_vm.pkl",'rb') as f:
    f1=pickle.load(f)
with open("good_params_sm.pkl",'rb') as f:
    f2=pickle.load(f)
with open("curriculum_vm.pkl",'rb') as f:
    f3=pickle.load(f)
with open("curriculum_sm.pkl",'rb') as f:
    f4=pickle.load(f)

pass