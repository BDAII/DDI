# Drug-drug interaction (DDI)
The proposed model consists of a graph convolutional autoEncoder network (GCAN) for embedding drug induced transcriptome data from the L1000 database of the Integrated Network-based Cellular Signatures (LINCS) project; and a long short-term memory network (LSTM) for DDI prediction. For a case study, we applied the proposed deep-learning model to antidiabetic agents.

## File directory structure
<div>
    <div style="float:left">"/data/":&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;folder of experiment data</div>
</div>

<div>
    <div style="float:left">"/data_preprocessing/GCAN/": use the GCAN framework for data preprocessing</div>
</div>

<div>
    <div style="float:left">"/model_train/GCAN_LSTM/" :&nbsp; the  GCAN + LSTM framework code</div>
</div>

<div>
    <div style="float:left">"/compare/": &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;&nbsp;six types of comparison model code </div>
</div>

<div>
    <div style="float:left">"/result/":&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp; folder of results</div>
</div>

## The required toolkit support and verified version number for project execution

| toolkit    | numpy  |pandas  |tensorflow |skmultilearn  |sklearn  |
| ----------| -------| -------| ----------| -------------| --------| 
| version number    | 1.16.3|1.0.3| 1.13.1| 0.2.0 | 0.21.2|

## Code instructions

1. data preprocessing

   `inits.py Layer.py Model.py` : initialization of the GCAN framework
   
   `train.py` : the GCAN framework is used for data preprocessing
    
2. model training
   
   `train_GEDDI_model.py param1 param2` : use GCAN + LSTM framework to train model. param1: train data; param2: test data

3. Compare the model

   `compare_model.py param1 param2`  : contrast models include "MLARAM", "MLkNN", "BRkNNa", "BRkNNb", "RF", "MLTSVM". param1: train data; param2: test data
  
