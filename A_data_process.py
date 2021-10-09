# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 00:11:56 2021

@author: lweike
"""

import pandas as pd
import numpy as np

from simpledbf import Dbf5

dbf = Dbf5('C:/Users/HUGUO/Desktop/TRB2022/crash_bicycle/PedBikeCrashes2007to2019/NCBike0719_OnRoadway.dbf') 

data = dbf.to_dataframe()
data["Level"]=data['RdCharacte']
data=data.drop_duplicates()
data.columns
data=data[['BikeAgeGrp','BikeSex','BikeAlcFlg','BikePos',
           'DrvrAgeGrp','DrvrSex','DrvrAlcFlg','SpeedLimit','DrvrVehTyp',
           'BikeDir','CrashLoc','Developmen','LightCond','Locality', 
           'NumLanes','RdCharacte','Level',
           'Weather', 'CrashHour','CrashDay','CrashMonth',
           'CrashGrp',
           'CrashSevr',]]

## data analysis
##
for col_name in data.columns:
    print("------------------------")
    print("the analysis of "+col_name)
    print(data[col_name].value_counts())
    
## print data analysis
def print_data_analysis(data,col_name):
    print("------------------------")
    print("the analysis of "+col_name)
    print(data[col_name].value_counts())    

##---------------------------------------------
'BikeAgeGrp','BikeSex','BikeAlcFlg','BikePos',
##---------------------------------------------

col_name="BikeAgeGrp"
print_data_analysis(data,col_name)

col_name="BikeSex"
print_data_analysis(data,col_name)

col_name="BikeAlcFlg"
data[col_name]=data[col_name].replace(["Missing","Unknown"],"Missing info") 
print_data_analysis(data,col_name)

col_name="BikePos"
data[col_name]=data[col_name].replace(["Bike Lane / Paved Shoulder"],"Bike Lane")    
data[col_name]=data[col_name].replace(["Sidewalk / Crosswalk / Driveway Crossing"],"Sidewalk")    
data[col_name]=data[col_name].replace(["Driveway / Alley"],"Driveway")  
data[col_name]=data[col_name].replace(["Unknown","Non-Roadway","Multi-use Path"],"Other")  
print_data_analysis(data,col_name)

##---------------------------------------------
'DrvrAgeGrp','DrvrSex','DrvrAlcFlg','SpeedLimit','DrvrVehTyp',
##---------------------------------------------

col_name='DrvrAgeGrp'
print_data_analysis(data,col_name)

col_name='DrvrSex'
print_data_analysis(data,col_name)

col_name='DrvrAlcFlg'
data[col_name]=data[col_name].replace(["Missing","Unknown"],"Missing info") 
print_data_analysis(data,col_name)

col_name='SpeedLimit'
print_data_analysis(data,col_name)

col_name='DrvrVehTyp'          #it is a question
data[col_name]=data[col_name].replace(["Passenger Car"],"Passenger car")  
data[col_name]=data[col_name].replace(["Pickup"],"Pick up")    
data[col_name]=data[col_name].replace(["Sport Utility"],"SUV")    
data[col_name]=data[col_name].replace(["Van"],"Van")  
data[col_name]=data[col_name].replace(["Light Truck (Mini-Van, Panel)",
                                        "Single Unit Truck (2-Axle, 6-Tire)",
                                        "Commercial Bus",
                                        "Single Unit Truck (3 Or More Axles)",
                                        "School Bus","Other Bus",
                                        "Unknown Heavy Truck",
                                        "Activity Bus",
                                        "Truck/Tractor","Truck or bus","Truck/Trailer",
                                        "Firetruck"],"Truck or bus")  
data[col_name]=data[col_name].replace(["Unknown","Motorcycle","Police",
                                       "Tractor/Semi-Trailer","Pedalcycle","Taxicab",
                                       "Moped","Unknown Heavy Truck","Pedestrian",
                                       "Motor Home/Recreational Vehicle",
                                       "EMS Vehicle, Ambulance, Rescue Squad",
                                       "All Terrain Vehicle (ATV)"],"Other/unknown")  
print_data_analysis(data,col_name)

##--------------------------------------
'BikeDir','CrashLoc','Developmen','LightCond','Locality', 
##--------------------------------------

col_name='BikeDir'
data[col_name]=data[col_name].replace(["Not Applicable","Unknown"],"Other or unknown") 
print_data_analysis(data,col_name)

col_name='CrashLoc'           #
print_data_analysis(data,col_name)

col_name='Developmen'
data[col_name]=data[col_name].replace(["Farms, Woods, Pastures","Institutional","Industrial"],"Other") 
print_data_analysis(data,col_name)

col_name='LightCond'
data[col_name]=data[col_name].replace(["Dusk","Dawn",'Dark - Unknown Lighting','Unknown'],"Other") 
print_data_analysis(data,col_name)

col_name='Locality'
print_data_analysis(data,col_name)

##--------------------------------------
'NumLanes','RdCharacte','Level',
##--------------------------------------

col_name='NumLanes'
data[col_name]=data[col_name].replace(["1 lane",'2 lanes'],"<=2 lanes") 
data[col_name]=data[col_name].replace(['3 lanes','4 lanes'],"3~4 lanes") 
data[col_name]=data[col_name].replace(['5 lanes','6 lanes','7 lanes','8 lanes','9 lanes','9 or more lanes'],">4 lanes") 
print_data_analysis(data,col_name)

col_name='RdCharacte'
data[col_name]=data[col_name].replace(["Straight - Level",'Straight - Grade',"Straight - Hillcrest","Straight - Bottom"],"Straight") 
data[col_name]=data[col_name].replace(["Curve - Level",'Curve - Grade',"Curve - Hillcrest","Curve - Bottom"],"Curve")
data[col_name]=data[col_name].replace(["Missing",'Unknown',"Other"],"Other/Unknown")
print_data_analysis(data,col_name)

col_name='Level'
data[col_name]=data[col_name].replace(["Straight - Level","Curve - Level"],"Level") 
data[col_name]=data[col_name].replace(['Straight - Grade','Curve - Grade',"Straight - Hillcrest","Curve - Hillcrest","Straight - Bottom","Curve - Bottom"],"Grade")
data[col_name]=data[col_name].replace(["Missing",'Unknown',"Other"],"Other/Unknown")
print_data_analysis(data,col_name)

##--------------------------------------
'Weather', 'CrashHour','CrashDay','CrashMonth',
##--------------------------------------

col_name='Weather'
data[col_name]=data[col_name].replace(["Clear","Cloudy"],"Clear or cloudy") 
data[col_name]=data[col_name].replace(["Rain","Fog, Smog, Smoke","Snow, Sleet, Hail, Freezing Rain/Drizzle","Fog - Smog - Smoke","Snow - Sleet -  Hail -  Freezing Rain/drizzle"],"Inclement") 
print_data_analysis(data,col_name)

col_name='CrashHour'
data[col_name]=data[col_name].replace([2,3,4,5,6],"Early morning") 
data[col_name]=data[col_name].replace([7,8,9,10],"Morning peak") 
data[col_name]=data[col_name].replace([11,12,13,14],"Mid-day") 
data[col_name]=data[col_name].replace([15,16,17,18],"Afternoon peak") 
data[col_name]=data[col_name].replace([19,20,21,22,23,0,1],"Afternoon peak") 
print_data_analysis(data,col_name)

col_name='CrashDay'
data[col_name]=data[col_name].replace(["Monday","Tuesday","Wednesday","Thursday","Friday"],"Weekday") 
data[col_name]=data[col_name].replace(["Saturday","Sunday"],"Weekend") 
print_data_analysis(data,col_name)

col_name='CrashMonth'
data[col_name]=data[col_name].replace(["March", "April", "May"],"Spring") 
data[col_name]=data[col_name].replace(["June", "July", "August"],"Summer") 
data[col_name]=data[col_name].replace(["September", "October", "November"],"Fall") 
data[col_name]=data[col_name].replace(["December", "January", "February"],"Winter") 
print_data_analysis(data,col_name)

##--------------------------------
'CrashGrp'
##--------------------------------
col_name='CrashGrp'

data[col_name]=data[col_name].replace(["Motorist Overtaking Bicyclist"],"Motorist overtaking")  
data[col_name]=data[col_name].replace(["Bicyclist Overtaking Motorist"],"Bicyclist overtaking")  
data[col_name]=data[col_name].replace(["Motorist Failed to Yield - Sign-Controlled Intersection",
                                       "Motorist Failed to Yield - Midblock",
                                       "Motorist Failed to Yield - Signalized Intersection"],"Motorist failed to yield") 
data[col_name]=data[col_name].replace(["Bicyclist Failed to Yield - Sign-Controlled Intersection",
                                       "Bicyclist Failed to Yield - Midblock",
                                       "Bicyclist Failed to Yield - Signalized Intersection"],"Bicyclist failed to yield") 
    
used_value_list=["Motorist overtaking","Bicyclist overtaking","Motorist failed to yield","Bicyclist failed to yield"]
other_value_list=list(set(list(data[col_name].value_counts().keys())).difference(set(used_value_list)))    

data[col_name]=data[col_name].replace(other_value_list,"Other") 
print_data_analysis(data,col_name)
##---------------------------------
'CrashSevr'
##---------------------------------
col_name='CrashSevr'
print_data_analysis(data,col_name)