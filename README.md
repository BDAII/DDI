# DDIS ʵ��  
���о���Ҫ�������ҩ���ת¼�����ݹ������Ӿ�ȷ��ҩ��-ҩ���໥����(drug�Cdrug interactions,DDIs)Ԥ��ģ���е�������Ҫ����:
1��ҩ��ת¼�������еĻ�������.
2������̽���˽�ҩ�￴��������Ϣ�Ƿ��������DDIs��Ԥ�⡣

## �ļ�Ŀ¼�ṹ  
<div>
    <div style="float:left">/data_preprocessing/GCN/</div>
    <div style="float:right">ʹ��GCN��ܽ�������Ԥ����Ĵ�����ļ���</div>
</div>

<div>
    <div style="float:left">/model_train/LSTM_GCN/</div>
    <div style="float:right">LSTM_GCN��ܴ��������ļ���</div>
</div>

<div>
    <div style="float:left">/compare/</div>
    <div style="float:right">���ֶԱ�ģ�ʹ�����ļ���</div>
</div>

<div>
    <div style="float:left">/result/</div>
    <div style="float:right">��������ַ</div>
</div>

## ��Ŀִ�����蹤�߰�֧�ּ���֤ͨ���İ汾��  

| ���߰�    | numpy  |pandas  |tensorflow |skmultilearn  |sklearn  |
| ----------| -------| -------| ----------| -------------| --------| 
| �汾��    | ������ |��������| 1.5| ������������ | ��������|

## ����˵��

1. ����Ԥ����

   `inits.py��Layer.py��Model.py`��;��GCN��ܵĳ�ʼ��
   
   `train.py`��;��GCN��ܽ�������Ԥ����
    
2. ģ��ѵ��
   
   `train_GEDDI_model.py param1 param2` ��;����LSTM+GCN���ѵ��GEDDIģ��, param1:ѵ�������ļ� ;param2: ���������ļ�

3. �Ա�ģ��

   `compare_model.py param1 param2` ��;�Ա�ģ���� "MLARAM", "MLkNN", "BRkNNa", "BRkNNb", "RF", "MLTSVM"����, param1:ѵ�������ļ� ;param2: ���������ļ�
    
    
    