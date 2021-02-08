# DDIS 实验  
本研究主要解决基于药物的转录组数据构建更加精确的药物-药物相互作用(drug–drug interactions,DDIs)预测模型中的两个主要问题:
1）药物转录组数据中的机器噪声.
2）本文探究了将药物看做序列信息是否更有利于DDIs的预测。

## 文件目录结构  
<div>
    <div style="float:left">/data_preprocessing/GCN/</div>
    <div style="float:right">使用GCN框架进行数据预处理的代码的文件夹</div>
</div>

<div>
    <div style="float:left">/model_train/LSTM_GCN/</div>
    <div style="float:right">LSTM_GCN框架代码所的文件夹</div>
</div>

<div>
    <div style="float:left">/compare/</div>
    <div style="float:right">六种对比模型代码的文件夹</div>
</div>

<div>
    <div style="float:left">/result/</div>
    <div style="float:right">结果保存地址</div>
</div>

## 项目执行所需工具包支持及验证通过的版本号  

| 工具包    | numpy  |pandas  |tensorflow |skmultilearn  |sklearn  |
| ----------| -------| -------| ----------| -------------| --------| 
| 版本号    | 。。。 |。。。。| 1.5| 。。。。。。 | 。。。。|

## 代码说明

1. 数据预处理

   `inits.py、Layer.py、Model.py`　;对GCN框架的初始化
   
   `train.py`　;用GCN框架进行数据预处理
    
2. 模型训练
   
   `train_GEDDI_model.py param1 param2` 　;采用LSTM+GCN框架训练GEDDI模型, param1:训练数据文件 ;param2: 测试数据文件

3. 对比模型

   `compare_model.py param1 param2` 　;对比模型有 "MLARAM", "MLkNN", "BRkNNa", "BRkNNb", "RF", "MLTSVM"六种, param1:训练数据文件 ;param2: 测试数据文件
    
    
    