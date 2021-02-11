# Drug-drug interaction (DDI)
The proposed model consisted of a Graph Convolutional AutoEncoder Network (GCAN) for embedding drug induced transcriptome data from the L1000 database of the Integrated Network-based Cellular Signatures (LINCS) project; and a Long Short-Term Memory network (LSTM) for DDI prediction. For a case study, we applied the proposed deep-learning model to antidiabetic agents.

## File directory structure
<div>
    <div style="float:left">/data_preprocessing/GCN/</div>
    <div style="float:right">use the GCN framework for data preprocessing</div>
</div>

<div>
    <div style="float:left">/model_train/LSTM_GCN/</div>
    <div style="float:right">The LSTM + GCN framework code</div>
</div>

<div>
    <div style="float:left">/compare/</div>
    <div style="float:right">six types of comparison model code</div>
</div>

<div>
    <div style="float:left">/result/</div>
    <div style="float:right">result folder</div>
</div>

## The required toolkit support and verified version number for project execution

| toolkit    | numpy  |pandas  |tensorflow |skmultilearn  |sklearn  |
| ----------| -------| -------| ----------| -------------| --------| 
| version number    | 1.16.3|1.0.3| 1.5| 0.2.0 | 0.21.2|

## Code instructions

1. data preprocessing

   `inits.py Layer.py Model.py` ;Initialization of the GCN framework
   
   `train.py` ;The GCN framework is used for data preprocessing
    
2. model training
   
   `train_GEDDI_model.py param1 param2` ; use LSTM+GCN framework to train GEDDI model, param1:train data ;param2: test data

3. Compare the model

   `compare_model.py param1 param2`  ;contrast models include "MLARAM", "MLkNN", "BRkNNa", "BRkNNb", "RF", "MLTSVM", param1:train data ;param2: test data
    
    
    
