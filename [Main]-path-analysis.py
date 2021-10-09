# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 00:23:47 2021

@author: lweike
"""

import A_data_process as my_data_process
import A_marginal_analysis as my_marginal_analysis
import numpy as np
import pandas as pd
import random 



#### read data
data=my_data_process.data
data=data[(data['CrashSevr']!="Unknown Injury") & (~data['CrashGrp'].isnull())]

data.head()
print(data.head())

##--------------------------------------------------------------------- variable
Contributing_factors=['BikeAgeGrp','BikeSex','BikeAlcFlg','BikePos',
           'DrvrAgeGrp','DrvrSex','DrvrAlcFlg','SpeedLimit','DrvrVehTyp',
           'BikeDir','CrashLoc','Developmen','LightCond','Locality', 
           'NumLanes','RdCharacte','Level',
           'Weather', 'CrashHour','CrashDay','CrashMonth']

Pre_crash_act=['CrashGrp']
Crash_severity=['CrashSevr']
continue_category=['BikeAgeGrp','DrvrAgeGrp','SpeedLimit','NumLanes']
##------------------------------------------------------------------------
col_name='BikeAgeGrp'
data=data[data[col_name]!='Unknown']
data_category= data[col_name].value_counts().keys().tolist()
#data_category=["Unknown",'0-5','6-10','11-15','16-19','20-24','25-29','30-39', '40-49','50-59','60-69','70+']
#replace_list=[0,1,1,2,2,3,3,4,5,6,7,8]
data_category=['0-5','6-10','11-15','16-19','20-24','25-29','30-39', '40-49','50-59','60-69','70+']
replace_list=[0,0,1,1,2,2,3,4,5,6,7]
data[col_name]=data[col_name].replace(data_category,replace_list)
my_data_process.print_data_analysis(data,col_name)

col_name='DrvrAgeGrp'
data_category= data[col_name].value_counts().keys().tolist()
data=data[data[col_name]!='Unknown']
#data_category=["Unknown","0-19","20-24","25-29","30-39","40-49","50-59","60-69","70+"]
#replace_list=[0,1,2,2,3,4,5,6,7]
data_category=["0-19","20-24","25-29","30-39","40-49","50-59","60-69","70+"]
replace_list=[0,1,1,2,3,4,5,6]
data[col_name]=data[col_name].replace(data_category,replace_list)
my_data_process.print_data_analysis(data,col_name)

col_name='SpeedLimit'
data_category= data[col_name].value_counts().keys().tolist()
data=data[data[col_name]!='Unknown']
#data_category=['Unknown','5 - 15 MPH','20 - 25  MPH','30 - 35  MPH','40 - 45  MPH','50 - 55  MPH',"60 - 75 MPH"]
#replace_list=[0,1,2,3,4,5,6]
data_category=['5 - 15 MPH','20 - 25  MPH','30 - 35  MPH','40 - 45  MPH','50 - 55  MPH',"60 - 75 MPH"]
replace_list=[0,1,2,3,4,5]
data[col_name]=data[col_name].replace(data_category,replace_list)
my_data_process.print_data_analysis(data,col_name)

col_name='NumLanes'
data_category= data[col_name].value_counts().keys().tolist()
data=data[data[col_name]!='Unknown']
#data_category=['Unknown','<=2 lanes', '3~4 lanes', '>4 lanes']
#replace_list=[0,1,2,3]
data_category=['<=2 lanes', '3~4 lanes', '>4 lanes']
replace_list=[0,1,2]
data[col_name]=data[col_name].replace(data_category,replace_list)
my_data_process.print_data_analysis(data,col_name)

##----Y_category
col_name='CrashSevr'
data_category= data[col_name].value_counts().keys().tolist()
data_category=['O: No Injury','C: Possible Injury','B: Suspected Minor Injury','A: Suspected Serious Injury','K: Killed']
replace_list=[0,1,2,3,4]
data[col_name]=data[col_name].replace(data_category,replace_list)
my_data_process.print_data_analysis(data,col_name)

##--data descriptions
print("------------------")
print("------------------")
data_descri={"var_name":[],
             "categories":[],
             "num":[]}
for col_name in  data.columns:  
    my_data_process.print_data_analysis(data,col_name)
    data_descri["var_name"].extend([col_name]*len(data[col_name].value_counts()))
    data_descri["categories"].extend(list(data[col_name].value_counts().index))
    data_descri["num"].extend(list(data[col_name].value_counts().values))

out_put_data=pd.DataFrame(data_descri)
out_put_data.to_csv("C:/Users/HUGUO/Desktop/TRB2022/crash_bicycle/"+"descriptions"+".csv")
##--------------------------------------------columns's variable process       

col_to_var={}
col_to_var_base={}

for col_name in data.columns:
    col_to_var[col_name]= data[col_name].value_counts().keys().tolist()
    
col_to_var_base={'BikeAgeGrp': 0,
 'BikeSex': 'Female',
 'BikeAlcFlg': 'No',
 'BikePos': 'Sidewalk',
 'DrvrAgeGrp': 0,
 'DrvrSex': 'Female',
 'DrvrAlcFlg':  'No',
 'SpeedLimit': 0,
 'DrvrVehTyp': 'Passenger car',
 'BikeDir': 'Facing Traffic',
 'Developmen': 'Other',
 'LightCond':  'Daylight',
 'Locality': 'Mixed (30% To 70% Developed)',
 'NumLanes': 0,
 'RdCharacte': 'Straight',
 'Level': 'Grade',
 'Weather': 'Other',
 'CrashLoc': 'Intersection',
 'CrashHour': 'Early morning',
 'CrashDay': 'Weekday',
 'CrashMonth': 'Spring', 
 'CrashGrp': 'Other',
 'CrashSevr':  0}

dummy_list=list(set(Contributing_factors+Pre_crash_act).difference(set(continue_category)))
data= data.dropna()


"input an value===========>0 or 1"
Other_var_dum= 1

if Other_var_dum==1:
    data_for_input_X= pd.get_dummies(data[dummy_list])
    data_for_input_X=my_marginal_analysis.delete_dummy_base(data_for_input_X,col_to_var_base)
Other_var_continue=1-Other_var_dum
if  Other_var_continue==1:
    data_for_input_X= my_marginal_analysis.label_coding(data,dummy_list)
    
for continue_var_col in continue_category:
        data_for_input_X[continue_var_col]=data[continue_var_col]
    
try:
    data_for_input_X=data_for_input_X.drop(columns=Crash_severity)
except KeyError:
    print("+========================================================")
    print("data_for_input_X don't have the columns of Crash_severity")
    print("+========================================================")
    
##-------get X and Y      
X = data_for_input_X.values
Y = data[Crash_severity].values


Y_list=data[Crash_severity[0]].value_counts().keys().tolist()
Y_list.sort()


X_test_column_order=dict(zip(data_for_input_X.columns.tolist(),list(range(0,len(data_for_input_X.columns.tolist())))))

##---------split data to ensure keep same in another related test
test_size=0.2
X_n=len(X)
X_test_index_list=[]

X_train= []
X_test=[]
y_train=[]
y_test=[]

for i in range(0,X_n):
    
    if random.random()<=test_size:
        X_test_index_list.append(i)
        
        X_test.append(X[i,:])
        y_test.append(Y[i,:])
    else:
        X_train.append(X[i,:])
        y_train.append(Y[i,:])

X_train=np.array(X_train)
X_test=np.array(X_test)
y_train=np.array(y_train)
y_test=np.array(y_test)

##------- split data random
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X,
#                                                    Y,
#                                                    test_size=0.2,
#                                                    random_state=1)
        
# Feature Scaling
from sklearn.preprocessing import StandardScaler

#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

import time 
start = time.time()

model="Boost"

if model=="RF":
    from sklearn.ensemble import RandomForestClassifier
    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(X_train,y_train.ravel())
    y_pred=clf.predict(X_test)
elif model=="CNB":
    from sklearn.naive_bayes import CategoricalNB
    clf = CategoricalNB()
    clf.fit(X_train, y_train.ravel())
    y_pred=clf.predict(X_test)
elif model=="SVM":    
    from sklearn import svm
    clf = svm.SVC(kernel='linear',probability=True)
    clf.fit(X_train, y_train.ravel())
    y_pred = clf.predict(X_test)
elif model=="Boost":  
    from sklearn.ensemble import AdaBoostClassifier
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, y_train.ravel())
    y_pred=clf.predict(X_test)
elif model=="NN":  
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(5, 4), max_iter=2000 ,random_state=1)
    clf.fit(X_train, y_train.ravel())
    y_pred=clf.predict(X_test)   

end = time.time()
print("X+Y1=Y2 Time =")
print(end-start)
##-----evaluation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print(model+" Accuracy:",metrics.accuracy_score(y_test, y_pred))

#feature_importances_dataframe = pd.DataFrame(clf.feature_importances_,index = data_for_input_X.columns,columns=['importance']).sort_values('importance',ascending=False)

var_marginal_dataframe, base_to_Y_probability =my_marginal_analysis.cal_marginal_effect(clf,
                        X_test, 
                        col_to_var,col_to_var_base,
                        continue_category,dummy_list,
                        X_test_column_order,Y_list)
var_marginal_dataframe.columns=["Var base","Var specific",
                                'O: No Injury','C: Possible Injury','B: Suspected Minor Injury','A: Suspected Serious Injury','K: Killed']
base_to_Y_probability.columns=["Var base","Var specific",
                               "U-"+'O: No Injury',"U-"+'C: Possible Injury',"U-"+'B: Suspected Minor Injury',"U-"+'A: Suspected Serious Injury',"U-"+'K: Killed']

X_Y1_Y2_new_data_for_CSV=pd.concat([var_marginal_dataframe, base_to_Y_probability ],axis=1)

X_Y1_Y2_new_data_for_CSV.to_csv("C:/Users/HUGUO/Desktop/TRB2022/crash_bicycle/"+model+"_X+Y1 to Y2"+ ".csv")


################################comparsion

"run A_main_X toY1.py"

import A_main_X_toY1 

other_analysis=A_main_X_toY1.get_X_y1_analysis_dataframe(X_test_index_list, model)

path_ana_value=["DirectMEonY2","uxForY1","MEonY1","uY1ForY2","MEofY1onY2","IndirectMEonY2","TotalMEonY2"]
Path_Analysis_table=pd.DataFrame("",columns=["Var base","Var specific"]+path_ana_value,
                                 index=X_Y1_Y2_new_data_for_CSV.index.tolist())

Path_Analysis_table["Var base"]=X_Y1_Y2_new_data_for_CSV["Var base"]
Path_Analysis_table["Var specific"]=X_Y1_Y2_new_data_for_CSV["Var specific"]

Path_Analysis_table["DirectMEonY2"]=X_Y1_Y2_new_data_for_CSV['B: Suspected Minor Injury']+\
                                    X_Y1_Y2_new_data_for_CSV['A: Suspected Serious Injury']+\
                                    X_Y1_Y2_new_data_for_CSV['K: Killed']
                                    
#------------------------------other_analysis is used here
Path_Analysis_table["uxForY1"]=other_analysis["U-"+'Bicyclist failed to yield']
Path_Analysis_table["MEonY1"]=other_analysis['Bicyclist failed to yield']

Path_Analysis_table["uY1ForY2"]=X_Y1_Y2_new_data_for_CSV["U-"+'B: Suspected Minor Injury']["CrashGrp@Bicyclist failed to yield"]+\
                                X_Y1_Y2_new_data_for_CSV["U-"+'A: Suspected Serious Injury']["CrashGrp@Bicyclist failed to yield"]+\
                                X_Y1_Y2_new_data_for_CSV["U-"+'K: Killed']["CrashGrp@Bicyclist failed to yield"]

Path_Analysis_table["MEofY1onY2"]=X_Y1_Y2_new_data_for_CSV['B: Suspected Minor Injury']["CrashGrp@Bicyclist failed to yield"]+\
                                X_Y1_Y2_new_data_for_CSV['A: Suspected Serious Injury']["CrashGrp@Bicyclist failed to yield"]+\
                                X_Y1_Y2_new_data_for_CSV['K: Killed']["CrashGrp@Bicyclist failed to yield"]
                                
Path_Analysis_table["IndirectMEonY2"]=(Path_Analysis_table["uxForY1"]+Path_Analysis_table["MEonY1"])*\
                                      (Path_Analysis_table["uY1ForY2"]+Path_Analysis_table["MEofY1onY2"])-\
                                      Path_Analysis_table["uxForY1"]*Path_Analysis_table["uY1ForY2"]
                                      
Path_Analysis_table["TotalMEonY2"]=Path_Analysis_table["DirectMEonY2"]+Path_Analysis_table["IndirectMEonY2"]

Path_Analysis_table.to_csv("C:/Users/HUGUO/Desktop/TRB2022/crash_bicycle/"+"path_analysis_"+model+".csv")