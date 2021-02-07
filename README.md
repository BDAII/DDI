邮件交流：。。。。
 

# DDIS 实验  
目的：。。。。  

## 文件目录结构  
<div>
    <div style="float:left">/data_preprocessing/GCN/</div>
    <div style="float:right">使用GCN框架进行数据预处理的代码的文件夹</div>
</div>

<div>
    <div style="float:left">/data_preprocessing/original_data/</div>
    <div style="float:right">本研究实验选择的原始数据集的文件夹</div>
</div>

<div>
    <div style="float:left">/data_preprocessing/feature_processed_data/</div>
    <div style="float:right">由原始数据进行特征处理后得到的用于模型训练的数据集的文件夹</div>
</div>

<div>
    <div style="float:left">/model_train/LSTM_GCN/</div>
    <div style="float:right">LSTM_GCN框架代码所的文件夹</div>
</div>

<div>
    <div style="float:left">/compare/</div>
    <div style="float:right">对比模型代码的文件夹</div>
</div>

<div>
    <div style="float:left">/compare/RF/</div>
    <div style="float:right">RF对比模型代码的文件夹</div>
</div>

## 项目执行所需工具包支持及验证通过的版本号  

| 工具包    | numpy  |pandas  |tensorflow |skmultilearn  |sklearn  |
| ----------| -------| -------| ----------| -------------| --------| 
| 版本号    | 。。。 |。。。。| 。。。。。| 。。。。。。 | 。。。。|

## 代码说明

1. 数据预处理

   `inits.py、Layer.py、Model.py`　;对GCN框架的初始化
   
   `train.py`　;用GCN框架进行数据预处理
   
    
2. 模型训练
   
   `RNN_2layer.py` 　;采用LSTM_GCN框架训练RNN模型

3. 对比模型

   `compare_model.py` 　;将 MLARAM, MLkNN, BRkNNaClassifier, BRkNNbClassifier四种模型进行对比
     
   `RF.py` 　;对比RF模型
    
## 结果文件  
* 分析以下结果文件，收集返回给BDAI（以KUMC为例）  
    * /。。。/。。。/。。。　　　xxx
    * /。。。/。。。/。。。　　 xxx 
    
    