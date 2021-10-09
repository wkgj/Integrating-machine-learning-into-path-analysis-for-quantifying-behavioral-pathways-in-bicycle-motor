# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 00:45:32 2021

@author: lweike
"""

#---------------------------------variable lab-coding or one-hot

import numpy as np;
import pandas as pd;

def label_coding(data,col_list):
    data_for_input_X=data[col_list].copy()
    for col_name in col_list:
        data_category= data_for_input_X[col_name].value_counts().keys().tolist()
        replace_list=list(range(0,len(data_category)))
        data_for_input_X[col_name]=data_for_input_X[col_name].replace(data_category,replace_list)
    return data_for_input_X

def delete_dummy_base(data_for_input_X,col_to_var_base):    
    base_list=[]
    for col_name in col_to_var_base.keys():
        base_list.append(col_name+"_"+str(col_to_var_base[col_name]))
        
    for col_name in data_for_input_X.columns:
        
        if col_name in base_list:
            print("delete "+ col_name)
            data_for_input_X=data_for_input_X.drop([col_name], axis=1)
    return data_for_input_X.copy()

def cal_marginal_effect(clf,                    # 训练好的分类模型
                        X_test,                 # 划分好的测试集数据，是一个多维数组，不需要标准化
                        col_to_var,             # 原数据集中每一列有哪些变量值，是个字典
                        col_to_var_base,        # 原数据集中每一列有哪些变量设为了base
                        continue_category,      # 连续变量（原数据列名）的集合，是个list
                        dummy_list,             # 需要虚拟的变量（原数据列名）的集合，是个list
                        X_test_column_order,    # Xtest测试集中，（Xtest数组列的order 对应哪个列名）
                        Y_list):                # Y的lab list，从小到大。
    
    marginal_var_index=[]
    marginal_var_base=[]
    marginal_var_specific=[]
    
    marignal_effets=Y_list
    
    for col_name in col_to_var.keys():
        
        for var in col_to_var[col_name]:
            if var != col_to_var_base[col_name]:
                marginal_var_index.append(col_name+"@"+str(var))
                marginal_var_base.append(col_to_var_base[col_name])
                marginal_var_specific.append(str(var))
    
    var_marginal_dataframe=pd.DataFrame("",columns=["Var base","Var specific"]+marignal_effets,index=marginal_var_index)
    var_marginal_dataframe["Var base"]=marginal_var_base
    var_marginal_dataframe["Var specific"]=marginal_var_specific
    
    base_to_Y_probability= var_marginal_dataframe.copy()
    
    ##--------------------------------------------for continue_category
    for col_name in continue_category:
        
        var_base=col_to_var_base[col_name]       # calculate base's probility 
        X_test_copy=X_test.copy()
        col_order=X_test_column_order[col_name];
        X_test_copy[:,col_order]=int(var_base)
        prob_y_2 = clf.predict_proba(X_test_copy)
        base_proba=np.mean(prob_y_2, axis=0)
        
        for var in col_to_var[col_name]:   
             if var != col_to_var_base[col_name]:
                 X_test_copy=X_test.copy()
                 col_order=X_test_column_order[col_name];
                 X_test_copy[:,col_order]=int(var)
                 prob_y_2 = clf.predict_proba(X_test_copy)
                 var_proba=np.mean(prob_y_2, axis=0)
                 
                 var_marginal=var_proba-base_proba
                 
                 var_index_in_dataframe=col_name+"@"+str(var)
                 
                 for col in marignal_effets:
                     
                     var_marginal_dataframe[col][var_index_in_dataframe]=var_marginal[col]
                     base_to_Y_probability[col][var_index_in_dataframe]=base_proba[col]
    
    for col_name in dummy_list:
        
        X_test_copy=X_test.copy()
        for var in col_to_var[col_name]: 
            if var != col_to_var_base[col_name]:                
                col_order=X_test_column_order[col_name+"_"+var]
                X_test_copy[:,col_order]=0
                prob_y_2 = clf.predict_proba(X_test_copy)
                base_proba=np.mean(prob_y_2, axis=0)   
        
       
        for var in col_to_var[col_name]: 
            
            if var != col_to_var_base[col_name]:
                X_test_for_dummy=X_test_copy.copy()      ## the value of all dummy of a variable is zeor  
                
                col_order=X_test_column_order[col_name+"_"+var]
                X_test_for_dummy[:,col_order]=1          ## change this dummy's value =1
                
                prob_y_2 = clf.predict_proba(X_test_for_dummy)
                
                var_proba=np.mean(prob_y_2, axis=0)
                 
                var_marginal=var_proba -base_proba  
                var_index_in_dataframe=col_name+"@"+str(var)
                for col in marignal_effets: 
                     var_marginal_dataframe[col][var_index_in_dataframe]=var_marginal[col] 
                     base_to_Y_probability[col][var_index_in_dataframe]=base_proba[col]
                     
    return var_marginal_dataframe.copy(),base_to_Y_probability.copy()