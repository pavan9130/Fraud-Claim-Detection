import pandas as pd
import numpy as np
import joblib
#importing data 
data=pd.read_csv('C:/Users/Pavan786/Downloads/InsuranceDataset.csv')
#dropping Hospital Id 
data=data.drop(['Hospital Id'],axis=1)
categorical_feature_list = ['Area_Service','Hospital County','Age','Gender','Cultural_group','ethnicity','Admission_type','Home or self care,','apr_drg_description','Code_illness','Mortality risk','Surg_Description','Abortion','Emergency dept_yes/No','Payment_Typology','Result']
data_cat=data[categorical_feature_list]
data_cat.head()
#taking numerical features 
num_features=[]
for col in data.columns:
    if col not in categorical_feature_list:
      num_features.append(col)

data_num = data[num_features]
data_cat=data_cat.drop(['Hospital County','Home or self care,','apr_drg_description'],axis=1)
# In 'Days_spend_hsptl'  value '120 +' 
#which causes issue while plotting 
#changing that to '120'
for i in range(0,len(data_num['Days_spend_hsptl'])):
     if data_num.iloc[i,0]=='120 +':
        data_num.iloc[i,0]='120'
data_num['Days_spend_hsptl']=pd.to_numeric(data_num['Days_spend_hsptl'],downcast='integer')
data_num=data_num.join(data['Result'])


# replacing NaN for Area_Service with mode
mode_Area_Service = data_cat['Area_Service'].mode().values[0]
data_cat['Area_Service'].replace(np.nan,mode_Area_Service,inplace=True)
# replacing NaN for Area_Service with mode
mode_Mortality_risk = data_cat['Mortality risk'].mode().values[0]
data_cat['Mortality risk'].replace(np.nan,mode_Mortality_risk,inplace=True)
licat=[]
licat=data_cat.columns.values.tolist()
licat.remove('Result')
licat
data_cat_df=pd.get_dummies(data_cat,columns=licat,drop_first=False)


#data_num=data_num.drop(['Result'],axis=1)
data_final = pd.concat([data_num,data_cat_df],axis=1)
selected_features_2=['Days_spend_hsptl',
 'ccs_diagnosis_code',
 'ccs_procedure_code',
 'Weight_baby',
 'Tot_charg',
 'Tot_cost',
 'ratio_of_total_costs_to_total_charges',
 'Area_Service_Capital/Adirond',
 'Area_Service_Central NY',
 'Area_Service_Finger Lakes',
 'Area_Service_Hudson Valley',
 'Area_Service_New York City',
 'Area_Service_Southern Tier',
 'Area_Service_Western NY',
 'Age_0 to 17',
 'Age_18 to 29',
 'Age_30 to 49',
 'Age_50 to 69',
 'Age_70 or Older',
 'Gender_F',
 'Gender_M',
 'Cultural_group_Black/African American',
 'Cultural_group_Other Race',
 'Cultural_group_White',
 'ethnicity_Not Span/Hispanic',
 'ethnicity_Spanish/Hispanic',
 'ethnicity_Unknown',
 'Admission_type_Elective',
 'Admission_type_Emergency',
 'Admission_type_Urgent',
 'Code_illness_1',
 'Code_illness_2',
 'Code_illness_3',
 'Code_illness_4',
 'Mortality risk_1.0',
 'Mortality risk_2.0',
 'Mortality risk_3.0',
 'Mortality risk_4.0',
 'Emergency dept_yes/No_N',
 'Emergency dept_yes/No_Y',
 'Payment_Typology_1',
 'Payment_Typology_2',
 'Payment_Typology_3',
 'Payment_Typology_4','Result']
data_featureselected_2=data_final[selected_features_2]
# transform the dataset
from imblearn.over_sampling import RandomOverSampler
over = RandomOverSampler(random_state=1)
data_featureselected_s=data_featureselected_2.sample(frac=0.01, random_state=1)
X_s, y_s = over.fit_resample(data_featureselected_s.iloc[:,0:44], data_featureselected_s.iloc[:,44])
from sklearn.model_selection import train_test_split
X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(X_s,y_s,test_size=0.20,random_state=120)
#RandomForestClassifier on over sampling dataset 2 
from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import classification_report
rfc1=RandomForestClassifier(random_state=122)
print("************model building*********")
rfc1.fit(X_train_o, y_train_o)

pkl_filename = "pickle_model.pkl"
with open(pkl_filename, 'wb') as file:
    joblib.dump(rfc1, file)       
            
#pickle.dump(rfc1,open('model.pkl','wb'))