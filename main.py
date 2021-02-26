import flask
from flask import  request
import pandas as pd
import joblib

app = flask.Flask(__name__, template_folder='Templates')
app.secret_key = 'pava123'
@app.route('/')
def main():
    return(flask.render_template('main.html'))

@app.route('/predict',methods=['POST'])
def predict():
       # formdata = session.get('formdata', None)
        Days_spend_hsptl = request.form.get('Days_spend_hsptl')  
        ccs_diagnosis_code = request.form.get('ccs_diagnosis_code')
        ccs_procedure_code = request.form.get('ccs_procedure_code')  
        Weight_baby = request.form.get('Weight_baby')
        Tot_charg = request.form.get('Tot_charg')
        Tot_cost = request.form.get('Tot_cost')  
        ratio = request.form.get('ratio_of_total_costs_to_total_charges')
        Area = request.form.get('Area') 
        Age = request.form.get('Age')
        Gender = request.form.get('Gender')
        Cultural = request.form.get('Cultural')  
        Admission = request.form.get('Admission')
        Mortality = request.form.get('Mortality')
        Ethnicity = request.form.get('Ethnicity')
        Emergency = request.form.get('Emergency')
        Codeillness = request.form.get('Codeillness')
        Payment = request.form.get('Payment')
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
          'Payment_Typology_4']
        dt={}
        dt[selected_features_2[0]]=Days_spend_hsptl
        dt[selected_features_2[1]]=ccs_diagnosis_code
        dt[selected_features_2[2]]=ccs_procedure_code
        dt[selected_features_2[3]]=Weight_baby
        dt[selected_features_2[4]]=Tot_charg
        dt[selected_features_2[5]]=Tot_cost
        dt[selected_features_2[6]]=ratio
        if Area=='Capital/Adirond':
             dt[selected_features_2[7]]=1
        else:
            dt[selected_features_2[7]]=0
        if Area=='Central NY':
             dt[selected_features_2[8]]=1
        else:
            dt[selected_features_2[8]]=0
        if Area=='Finger Lakes':
             dt[selected_features_2[9]]=1
        else:
            dt[selected_features_2[9]]=0
        if Area=='Hudson Valley':
             dt[selected_features_2[10]]=1
        else:
            dt[selected_features_2[10]]=0
        if Area=='New York City':
             dt[selected_features_2[11]]=1
        else:
            dt[selected_features_2[11]]=0
        if Area=='Southern Tier':
             dt[selected_features_2[12]]=1
        else:
            dt[selected_features_2[12]]=0
        if Area=='Western NY':
             dt[selected_features_2[13]]=1
        else:
            dt[selected_features_2[13]]=0
       
            
            
        if Age=='0 to 17':
             dt[selected_features_2[14]]=1
        else:
            dt[selected_features_2[14]]=0
        if Age=='18 to 29':
             dt[selected_features_2[15]]=1
        else:
            dt[selected_features_2[15]]=0
            
        if Age=='30 to 49':
             dt[selected_features_2[16]]=1
        else:
            dt[selected_features_2[16]]=0
            
        if Age=='50 to 69':
             dt[selected_features_2[17]]=1
        else:
            dt[selected_features_2[17]]=0    
                       
        if Age=='70 or Older':
             dt[selected_features_2[18]]=1
        else:
            dt[selected_features_2[18]]=0   

        if Gender=='F':
             dt[selected_features_2[19]]=1
        else:
            dt[selected_features_2[19]]=0   
        
        if Gender=='M':
             dt[selected_features_2[20]]=1
        else:
            dt[selected_features_2[20]]=0   
            
        if Cultural=='Black/African American':
             dt[selected_features_2[21]]=1
        else:
            dt[selected_features_2[21]]=0       
        if Cultural =='Other Race':
             dt[selected_features_2[22]]=1
        else:
            dt[selected_features_2[22]]=0       
        if Cultural =='White':
             dt[selected_features_2[23]]=1
        else:
            dt[selected_features_2[23]]=0       
        

        if Ethnicity=='Not Span/Hispanic':
             dt[selected_features_2[24]]=1
        else:
            dt[selected_features_2[24]]=0       
        if Ethnicity =='Spanish/Hispanic':
             dt[selected_features_2[25]]=1
        else:
            dt[selected_features_2[25]]=0       
        if Ethnicity =='Unknown':
             dt[selected_features_2[26]]=1
        else:
            dt[selected_features_2[26]]=0       
                              
        if Admission=='Elective':
             dt[selected_features_2[27]]=1
        else:
            dt[selected_features_2[27]]=0       
        if Admission =='Emergency':
             dt[selected_features_2[28]]=1
        else:
            dt[selected_features_2[28]]=0       
        if Admission =='Urgent':
             dt[selected_features_2[29]]=1
        else:
            dt[selected_features_2[29]]=0       
         
        
            
            
            
        if Codeillness=='1':
             dt[selected_features_2[30]]=1
        else:
            dt[selected_features_2[30]]=0       
        if Codeillness =='2':
             dt[selected_features_2[31]]=1
        else:
            dt[selected_features_2[31]]=0       
        if Codeillness =='3':
             dt[selected_features_2[32]]=1
        else:
            dt[selected_features_2[32]]=0 
            
        if Codeillness=='4':
             dt[selected_features_2[33]]=1
        else:
            dt[selected_features_2[33]]=0      
            
        
        if Mortality=='1':
             dt[selected_features_2[34]]=1
        else:
            dt[selected_features_2[34]]=0       
        if Mortality =='2':
             dt[selected_features_2[35]]=1
        else:
            dt[selected_features_2[35]]=0       
        if Mortality =='3':
             dt[selected_features_2[36]]=1
        else:
            dt[selected_features_2[36]]=0 
            
        if Mortality=='4':
             dt[selected_features_2[37]]=1
        else:
            dt[selected_features_2[37]]=0            
        
        if Emergency =='No':
             dt[selected_features_2[38]]=1
        else:
            dt[selected_features_2[38]]=0 
            
        if Emergency=='Yes':
             dt[selected_features_2[39]]=1
        else:
            dt[selected_features_2[39]]=0
         
        if Payment=='1':
             dt[selected_features_2[40]]=1
        else:
            dt[selected_features_2[40]]=0       
        if Payment =='2':
             dt[selected_features_2[41]]=1
        else:
            dt[selected_features_2[41]]=0       
        if Payment =='3':
             dt[selected_features_2[42]]=1
        else:
            dt[selected_features_2[42]]=0 
            
        if Payment=='4':
             dt[selected_features_2[43]]=1
        else:
            dt[selected_features_2[43]]=0
                  
        df=pd.DataFrame(dt, index=[0])
        with open("pickle_model.pkl", 'rb') as file:
          pickle_model = joblib.load(file)
        prediction=pickle_model.predict(df.iloc[:,:])
        output=prediction[0]
        
        #session.pop('formdata')
        if output==1:
             text='Insurance Claim is Fraud'
        else:
             text='Insurance Claim is Genuine'
        return(flask.render_template('main.html',prediction_text='{}'
                                     .format(text + str(output))))
                                                                                                
if __name__ == '__main__':
 app.run(debug=True,use_reloader=False)