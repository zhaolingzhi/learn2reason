# Learn2reason
Supporting materials for paper L2R

## Propositional Logic parsing as an demo
To generate a perfect model for evaluating propositional logic. We first train the value model.
`
python tf_recurnn_vm.py
`
The value model will be trained on `curriculum_vm.pkl` and text version of curriculum for value model is `curriculum_vm.txt` and then we get the value model. An example trained value model is `good_params_vm.pkl`. After we get the value model, we then train the score model by
`
python tf_recurnn_sm.py
`
This code uses the curriculum for score model `curriculum_sm.pkl` and text version of curriculum for score model is `curriculum_sm.txt`. The score model training also depends on the value model. Here the default value model used here is `good_params_vm.pkl`. After the program terminates, we will get a well-trained score model.  An example trained score model is `good_params_sm.pkl`. We can test the value model and score model by
`
python tf_recurnn_test.py
`
The default models tested are `good_params_vm.pkl` and `good_params_sm.pkl`. The test data is `logic_test.pkl` and text version is `logic_test.txt`. This shows our model runs 100% accurate on test data which are on average over 150 tokens long.


## Network Protocol Reverse Engineering task
In this section, we showcase learning a simple network protocol grammar through network messages. The network protocol to learn is the toy network protocol in [netzob tutorial](http://blog.amossys.fr/How_to_reverse_unknown_protocols_using_Netzob.html). The example network messages
are shown below.
````
'CMDidentify#\x07\x00\x00\x00Roberto'                
'RESidentify#\x00\x00\x00\x00\x00\x00\x00\x00'       
'CMDinfo#\x00\x00\x00\x00'                           
'RESinfo#\x00\x00\x00\x00\x04\x00\x00\x00info'       
'CMDstats#\x00\x00\x00\x00'                          
'RESstats#\x00\x00\x00\x00\x05\x00\x00\x00stats'     
'CMDauthentify#\n\x00\x00\x00aStrongPwd'             
'RESauthentify#\x00\x00\x00\x00\x00\x00\x00\x00'     
'CMDencrypt#\x06\x00\x00\x00abcdef'                  
"RESencrypt#\x00\x00\x00\x00\x06\x00\x00\x00$ !&'$"  
"CMDdecrypt#\x06\x00\x00\x00$ !&'$"                  
'RESdecrypt#\x00\x00\x00\x00\x06\x00\x00\x00abcdef'  
'CMDbye#\x00\x00\x00\x00'                            
...       
````
The message format of this network protocol does not contain recursive structure. We only need to learn a correct value model. To learn the format of this network message, we first use binary part as delimiter and then characterize the tokens that often appears to be special tokens and tokens that appear only once to be common tokens. Then we can use our method to learn the message format of this toy network messages. The learned model can achieve 100% accuracy. 
The text version of testing data is shown in `models/network_protocol/testing.txt`
The learned model is in the folder `models/network_protocol`. You can use `models/network_protocol/tf_recurnn_rl_test.py` to test the trained model. 

## Grep Input Format task
In this section, we show the preliminary result of using our model to infer the input format of linux program (Grep)[https://www.gnu.org/software/grep/doc/grep.html]. Grep is a pattern matching program which tries to find which part of a text matches a specific pattern. The format we are trying to infer is the format of *pattern*. The pattern is essentially a regular expressions. For example,
````
grep [ab] GPL-1
````
`[ab]` is the pattern and a correct pattern grep will try to find whether there is a or b in GPL-1. 
````
grep a[ GPL-1
```` 
`a[` is the pattern but an incorrect one. Grep will terminate and return that `a[` is an ill-formatted pattern. Our model try to infer the correct pattern format based on different inputs and what grep returns.
Given an input pattern, our model 
can decide whether this input is an legal pattern to the grep program or not. Preliminary results show that the trained model can achieve 100% accuracy on sequences with length 4 or less.
The text version of testing data is shown in `models/grep/testing.txt`.
The learned model is in the folder `models/grep`. You can use `models/grep/tf_recurnn_rl_test.py` to test the trained model. 

## OpenAI tasks 

In this section, we provide more details about OpenAI tasks and their curriculum.
The OpenAI tasks are adapted from our [OpenAI environment](https://github.com/openai/gym). The 
adapted environment is shown in the sub-module [gym](https://github.com/learn2reason/gym/tree/46fa86d5e5f26a78751afcef12a9d5b8b2723328). The basic setting  
is that we have an input tape and an output tape. The input tape provides access to input data symbols stored in 
an 'infinite' 1-D tape. A read head accesses one single character at a time. The output tape is similar to input
tape except there is a write head that writes one character at a time.  The illustration of different tasks is shown in `tasks.pdf`.

![Tasks](tasks.pdf)

### Copy task 

The copy task is a task that copies inputs from a tape. There is an no-op in vocabulary which stands for output
nothing. The model learns whether to move the reading head left or right and what to output. The default copy 
environment is the file in `gym/gym/envs/algorithmic/copy_.py` and the *curriculum* is in the file `gym/gym/envs/algorithmic/copy_c.py`. To use the default environment, we need to call environment. Refer to sub-module [gym](https://github.com/learn2reason/gym/tree/46fa86d5e5f26a78751afcef12a9d5b8b2723328) and the file `models/copy/custom_copy.py` to see how to apply the environment. 'Copy-v0' stands for the default environment and 'Copy-v2' stands for the curriculum environment.

### Duplicate Input task 

The duplicate input task is a task that input tape has duplicates every other character and the model learns to remove the duplicates. There is an no-op in vocabulary which stands for output
nothing. The model learns what to output. The default duplicate input 
environment is the file in `gym/gym/envs/algorithmic/duplicate_input.py` and the *curriculum* is in the file `gym/gym/envs/algorithmic/duplicate_input_c.py`. To use the default environment, we need to call environment. Refer to sub-module [gym](https://github.com/learn2reason/gym/tree/46fa86d5e5f26a78751afcef12a9d5b8b2723328) and the file `models/di/custom_di.py` to see how to apply the environment. 'DuplicatedInput-v0' stands for the default environment and 'DuplicatedInput-v2' stands for the curriculum environment.

### Reverse Addition task 

The reverse addition task is a task that input tape has two rows of input and the model learns to do the sum of the two rows of input. Since the input tape is read from left to right and human do addition from right to left, this is called reverse addition. The default reverse addition
environment is the file in `gym/gym/envs/algorithmic/reverse_addition.py` and the *curriculum* is in the file `gym/gym/envs/algorithmic/reverse_addition_c.py`. To use the default environment, we need to call environment. Refer to sub-module [gym](https://github.com/learn2reason/gym/tree/46fa86d5e5f26a78751afcef12a9d5b8b2723328) and the file `models/radd/custom_radd.py` to see how to apply the environment. 'ReversedAddition-v0' stands for the default environment and 'ReversedAddition-v2' stands for the curriculum environment.

### Single Multiplication task 

The single multiplication task is a task that input tape has one rows of input and a single digits and the model learns to do the multiplication of the row of input and single digit.  The default single multiplication
environment is the file in `gym/gym/envs/algorithmic/single_multiplication.py` and the *curriculum* is in the file `gym/gym/envs/algorithmic/single_multiplication_c.py`. To use the default environment, we need to call environment. Refer to sub-module [gym](https://github.com/learn2reason/gym/tree/46fa86d5e5f26a78751afcef12a9d5b8b2723328) and the file `models/sm/custom_sm.py` to see how to apply the environment. 'SingleMultiplication-v0' stands for the default environment and 'SingleMultiplication-v2' stands for the curriculum environment.


## Learned models

The learned models are in folder `models`


