#!/usr/bin/env python
# coding: utf-8

# In[187]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix


# In[188]:


dats=pd.read_csv(r"C:\Users\Infra\Desktop\faraz khan - project1_dataset.csv")
dats.head()


# In[190]:


# removing some NA values
mean_person_emp_length=dats["person_emp_length"].mean()
mean_loan_int_rate=dats["loan_int_rate"].mean()

dats["person_emp_length"].fillna(mean_person_emp_length,inplace=True)
dats["loan_int_rate"].fillna(mean_loan_int_rate,inplace=True)


# In[191]:


# converting string values to numerical values so it can processed by scaler() function

mapping_homeownership={"OWN":1,"RENT":2,"MORTGAGE":3,"OTHER":4}
mapping_loanintent={"PERSONAL":1,"EDUCATION":2,"MEDICAL":3,"VENTURE":4,"HOMEIMPROVEMENT":5,"DEBTCONSOLIDATION":6}
mapping_loanGrade={"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7}
mapping_default_file={"Y":1,"N":0}

# replacing in actual dataset
dats["person_home_ownership"]=dats["person_home_ownership"].replace(mapping_homeownership)
dats["loan_intent"]=dats["loan_intent"].replace(mapping_loanintent)
dats["loan_grade"]=dats["loan_grade"].replace(mapping_loanGrade)
dats["cb_person_default_on_file"]=dats["cb_person_default_on_file"].replace(mapping_default_file)


# In[82]:





# In[192]:


x=dats[["person_age","person_income","person_home_ownership","person_emp_length","loan_intent","loan_grade","loan_amnt","loan_int_rate","loan_percent_income","cb_person_default_on_file","cb_person_cred_hist_length"]]
y=dats["loan_status"]


# In[193]:


x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.3, random_state=42)


# In[194]:


scaler=StandardScaler()
x_train_scaler=scaler.fit_transform(x_train)
x_test_scaler=scaler.transform(x_test)


# In[195]:


model=LogisticRegression()


# In[196]:


model.fit(x_train_scaler,y_train)


# In[197]:


y_pred = model.predict(x_test_scaler)


# In[198]:


accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(report)
print(confusion)

