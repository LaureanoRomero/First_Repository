#!/usr/bin/env python
# coding: utf-8

# 
# Programa Machine Learning Fintech
# =============

# - Programa Creado por: Laureano Romero Velasquez.
# 
# - https://www.linkedin.com/in/laureanoromero/
# 
# - https://consultoriaestadistica.blogspot.com/

# **Parte 1 : Importación y creación de las librerias.**
# 
# - Se coloca las librerias que son necesarias para la creación de los algoritmos en AWS.

# In[204]:


import dateutil.parser
import pandas 
import numpy as np
import seaborn as sns

import xlsxwriter
import matplotlib.pyplot as plt
from sklearn.externals import joblib 
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, roc_auc_score,classification_report
from sklearn.metrics import precision_score
from sklearn.tree import export_graphviz

#from sklearn.datasets import load_iris, load_breast_cancer
#import graphviz
#plt.style.use("ggplot")


# **Parte 2 : Descripción del DataSet de  Fintech.**
# 
# - Para que el algoritmo de Machine Learning, funcione correctamente es fundamental que se carguen las mismas variables con el mismo formato con el cual se realizó el entrenamiento.

# |Variables|	Descripción                                           |
# |---------|-------------------------------------------------------|
# |Nombre  |Apellido del cliente  |
# |Sexo	  |Genero del cliente   |
# |N° Credito	  |Número del crédito|
# |sucursal	  |Sucursal|
# |Monto Credito	  |Valor del crédito|
# |Saldo   |Saldo del crédito|
# |Fecha De Pago	|Fecha de pago del crédito|
# |Fecha Desembolso	|Fecha de desembolso|
# |Valor cuota	|Valor de la cuota mensual|
# |Valor hoy 	 |Deuda a hoy, del crédito| 
# |Capital vencido	|Valor del capital vencido|
# |Interes vencido	|Valor del interes vencido|
# |Mora	|Valor de la mora Vencido|
# |Total vencido	|Total dinero vencido|
# |0-30 dias 	 |Valor vencido a 30 días.| 
# |31-60 dias	|Valor vencido a 60 días|
# |61-90 dias	|Valor vencido a 90 días|
# |91-120 dias	|Valor vencido a 120 días|
# |121-150 dias	|Valor vencido a 150 días|
# |151-180 dias	 |Valor vencido a 180 días| 
# |Mayor 180 dias 	 |Valor vencido a mas de 180 días| 
# |Ciudad	|Ciudad desembolso de credito|
# |Tipo credito	|Tipo de crédito|
# |Calificación	|Calificación del crédito|
# |Dias de mora	|días de mora del crédito|
# |Num.Cuotas Vencidas	|Numero de cuotas vencidas|
# |Fecha.Nac	|Fecha nacimiento|
# |Capita NO Venc.	|Capital que no esta vencido|
# |score	 |Calificación central de riesgo| 
# |edad 	 |Edad del cliente| 
# |Ingresos 	 |Ingresos del cliente| 
# |Actividad 	 |Actividad economica| 

# **Parte 3 : Objetivo Empresarial:** 
# 
# - Identificar la probabilidad de Impago por calificación de mejor a menor cliente (AA, A, BB, B, CC, C).
# 

# **parte 4 : Importación,almacenamiento de los dataset, manejo del titulo de las variables.**
# 
# - Se debe almacenar el Data Set, en la raiz de la documentación.

# In[205]:


#import mpld3
# Import excel
anno="2020"
mes = "04"
dia = "21"
Fintechdiamesanno = pandas.read_excel(f'BDFintech{dia}{mes}{anno}.xlsx', sheet_name='Sheet1')
#Renombrando el nombre de las variables, eliminando espacios y caracteres especiales.
# Data Fintech
Fintechdiamesanno1 = Fintechdiamesanno.rename(columns= {"N° Credito":"NCredito",
                                                        "Monto Credito":"MontoCredito",
                                                        "Fecha De Pago":"FechaDePago",
                                                         "Fecha Desembolso":"FechaDesembolso",   
                                                        "Valor cuota":"Valorcuota",
                                                        "Valor hoy":"Valorhoy",
                                                        "Capital vencido":"Capitalvencido",
                                                        "Interes vencido":"Interesvencido",
                                                        "0-30 dias":"_0_30_dias",
                                                        "31-60 dias":"_31_60_dias",
                                                        "61-90 dias":"_61_90_dias",
                                                        "91-120 dias":"_91_120_dias",
                                                        "121-150 dias":"_121_150_dias",
                                                        "151-180 dias":"_151_180_dias",
                                                        "Mayor 180 dias":"_Mayor_180_dias",
                                                        "Tipo credito":"Tipo_credito",
                                                        "Calificación":"Calificacion",
                                                        "Dias de mora":"Dias_de_mora",
                                                        "Num.Cuotas Vencidas":"Num_Cuotas_Vencidas",
                                                        "Fecha.Nac":"Fecha_Nac",
                                                        "Capita NO Venc.":"Capita_NO_Venc",
                                                        "Total vencido":"Totalvencido"})
Fintechdiamesanno1.iloc[0:3]


# In[206]:


from pandas import DataFrame

Fintechdiamesanno1.loc[Fintechdiamesanno1['Dias_de_mora'] <= 30, 'Impago'] = 0 
Fintechdiamesanno1.loc[Fintechdiamesanno1['Dias_de_mora'] > 30, 'Impago'] = 1 

Fintechdiamesanno1.iloc[0:3]


# **Parte 5 : Analisis Descriptivo del dataset.**
# 
# - Es importante conservar las propiedades de las variables originales. 

# In[207]:


#Ver tipo de variables del data set.
Fintechdiamesanno1.dtypes


# **Paso 6 : Eliminación de variables Irrelevantes para el pronóstico.**
# 
# - En este paso se procede a eliminar las variables que lo aportan a la predicción.
# - Mas de 100 categorias.

# In[208]:


#Coopdiamesanno1.drop(["F_INICIOFI","F_CARGOPRE",'SCORE',"ANNO", "SALDO_CAP_2020_1","SALDO_PROVI_2020_1"],axis=1)
del Fintechdiamesanno1["Nombre"] # + 100 categorias
del Fintechdiamesanno1["NCredito"] # + 100 categorias
del Fintechdiamesanno1["sucursal"] # es una costante = 1.
del Fintechdiamesanno1["FechaDePago"] # Fecha de pago, es una variable que ocurre a partir del evento.
del Fintechdiamesanno1["FechaDesembolso"] # Fecha de desembolso es una variable que ocurre a partir del evento.
del Fintechdiamesanno1["Mora"] # Es una constante =0
del Fintechdiamesanno1["Totalvencido"]  # es una variable que se da a partir del evento 
del Fintechdiamesanno1["_0_30_dias"] # es una variable que se da a partir del evento
del Fintechdiamesanno1["_31_60_dias"] # es una variable que se da a partir del evento
del Fintechdiamesanno1["_61_90_dias"] # es una variable que se da a partir del evento
del Fintechdiamesanno1["_91_120_dias"] # es una variable que se da a partir del evento
del Fintechdiamesanno1["_121_150_dias"] # es una variable que se da a partir del evento
del Fintechdiamesanno1["_151_180_dias"] # es una variable que se da a partir del evento
del Fintechdiamesanno1["_Mayor_180_dias"] # es una variable que se da a partir del evento.
del Fintechdiamesanno1["Dias_de_mora"] # es una variable que se da a partir del evento.
del Fintechdiamesanno1["Capita_NO_Venc"] # es una variable que se da a partir del evento.
del Fintechdiamesanno1["Interesvencido"] # Con el campo edad se representa esta variable.
del Fintechdiamesanno1["Capitalvencido"] # Con el campo edad se representa esta variable.
del Fintechdiamesanno1["Fecha_Nac"] # Con el campo edad se representa esta variable.
del Fintechdiamesanno1["Num_Cuotas_Vencidas"] # Con el campo edad se representa esta variable.
del Fintechdiamesanno1["Calificacion"] # Con el campo edad se representa esta variable.
del Fintechdiamesanno1["Valorhoy"] # Este campo se presenta cuando hay mora.
del Fintechdiamesanno1["Saldo"] # Este campo se origina despues de otorgar el crédito.


# In[209]:


Fintechdiamesanno1.columns


# **Analisis Descriptivo de frecuencias del dataset.**
# 
# - Es importante conservar las propiedades de las variables originales. 

# In[210]:


#Estadisticas Descriptivas del data set variables Categoricas.
print(Fintechdiamesanno1.describe(include=np.object))


# In[211]:


#Estadisticas Descriptivas del data set variables Númericas.
print(Fintechdiamesanno1.describe())


# In[212]:


f, ax = plt.subplots(figsize=(10, 8))
corr = Fintechdiamesanno1.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
               square=True, ax=ax)


# **Parte 6 : Limpieza del Data Set.**

# - En la variable **Ingresos** se cambia los ceros por valores vacios.
# 

# In[213]:


#Limpieza de los valores.
# Se reemplazan los valores ceros por vacios
Fintechdiamesanno1[['score', "Ingresos"]] = Fintechdiamesanno1[[
                                                                                    
                                                                                     'score', "Ingresos"]].replace(0,np.NaN)
#Imputación MontoCredito
Fintechdiamesanno1["MontoCredito"] = Fintechdiamesanno1["MontoCredito"].replace(np.NaN,Fintechdiamesanno1["MontoCredito"].median())
#Fintechdiamesanno1["Valorhoy"] = Fintechdiamesanno1["Valorhoy"].replace(np.NaN,0)
Fintechdiamesanno1["Ingresos"] = Fintechdiamesanno1["Ingresos"].replace(np.NaN,Fintechdiamesanno1["Ingresos"].median())
Fintechdiamesanno1["score"] = Fintechdiamesanno1["score"].replace(np.NaN,Fintechdiamesanno1["score"].median())
Fintechdiamesanno1["Actividad"] = Fintechdiamesanno1["Actividad"].replace(np.NaN,"INDEPENDIENTE")
#Fintechdiamesanno1["Valorcuota"] = Fintechdiamesanno1.Valorcuota * Fintechdiamesanno1.Valorcuota


# **Paso 6b : Comprobación de la depuración de la variable.**
# 
# - Se comprueba alguno de los cambios hechos. 

# In[214]:


#print(Test)
#Test.iloc[5:15]
Fintechdiamesanno1.loc[5:30, [ 'Impago']] # Columnas tip, sex y day


# **Paso 7 : Balanceo de la muestra.** 
# 
# - Aca es importante que cada vez que se entrene el modelo, Se deben de ajustar los tamaños de muestra, para los impagos, en este caso es 119 impagos. 

# In[215]:


df = pandas.DataFrame(np.random.normal(0, 1, (5, 2)), columns=["A", "B"])
df


# In[216]:


Fintechdiamesanno1.groupby('Impago').size()


# In[217]:


#Create customers bad data in python
clientes_malos = Fintechdiamesanno1.loc[Fintechdiamesanno1['Impago'] == 1]
display(clientes_malos[:5])
#Create customers god data in Python
Clientes_buenos = Fintechdiamesanno1.loc[Fintechdiamesanno1['Impago'] == 0]
display(Clientes_buenos[:5])
#Random n customers god
buenos_n=Clientes_buenos.sample(n=clientes_malos['Impago'].count(), random_state=12345678)
#Apilo los dos data set
Fintech_Balance=buenos_n.append(clientes_malos)
# Cantidad de filas y columnas
print(Fintech_Balance.shape)


# **Parte 8 : Exploración de los datos.** 
# 
# Personas Edades de personas que no pagan Vs Si pagan.
# 
# - Impresión de clientes que no pagan. 

# In[218]:


# Del grupo que no paga.
Fintech_Balance[Fintech_Balance.Impago==1].head()


# - **Distribución de los ingresos Personas que no pagan**

# In[219]:


Histog_Ingresos_nopaga = plt.hist(Fintech_Balance[Fintech_Balance.Impago==1].Ingresos.values, 10, facecolor="red", alpha=0.5)
plt.title("Distribición de Ingresos que no pagan")


# - **Distribución de Ingresos de personas que si pagan**

# In[220]:


Histog_Ingresos_sipaga = plt.hist(Fintech_Balance[Fintech_Balance.Impago==0].Ingresos.values, 10, facecolor="green", alpha=0.5)
plt.title("Distribición de Ingresos que si pagan")


# - **Distribución de edades Personas que no pagan**

# In[221]:


Histog_edad_nopaga = plt.hist(Fintech_Balance[Fintech_Balance.Impago==1].edad.values, 10, facecolor="red", alpha=0.5)
plt.title("Distribición de edades que no pagan")


# - **Distribución de edades de personas que si pagan**

# In[222]:


Histog_Edad_sipaga = plt.hist(Fintech_Balance[Fintech_Balance.Impago==0].edad.values, 10, facecolor="green", alpha=0.5)
plt.title("Distribución de edades que si pagan")


# - **Distribución de Score Personas que no pagan**

# In[223]:


Histog_score_nopaga = plt.hist(Fintech_Balance[Fintech_Balance.Impago==1].score.values, 10, facecolor="red", alpha=0.5)
plt.title("Distribución de Score que no pagan")


# - **Distribución de Score de personas que si pagan**

# In[224]:


Histog_score_sipaga = plt.hist(Fintech_Balance[Fintech_Balance.Impago==0].score.values, 10, facecolor="green", alpha=0.5)
plt.title("Distribución de Score que si pagan")


# - **Distribución de MontoCredito Personas que no pagan**

# In[225]:


Histog_MontoCredito_nopaga = plt.hist(Fintech_Balance[Fintech_Balance.Impago==1].MontoCredito.values, 10, facecolor="red", alpha=0.5)
plt.title("Distribución de MontoCredito que no pagan")


# - **Distribución de MontoCredito de personas que si pagan**

# In[226]:


Histog_MontoCredito_sipaga = plt.hist(Fintech_Balance[Fintech_Balance.Impago==0].MontoCredito.values, 10, facecolor="green", alpha=0.5)
plt.title("Distribución de MontoCredito que si pagan")


# - **Distribución de Saldo Personas que no pagan**

# In[227]:


#Histog_Saldo_nopaga = plt.hist(Fintech_Balance[Fintech_Balance.Impago==1].Saldo.values, 10, facecolor="red", alpha=0.5)
#plt.title("Distribución de Saldo que no pagan")


# - **Distribución de Saldo de personas que si pagan**

# In[228]:


#Histog_Saldo_sipaga = plt.hist(Fintech_Balance[Fintech_Balance.Impago==0].Saldo.values, 10, facecolor="green", alpha=0.5)
#plt.title("Distribución de Score que si pagan")


# - **Distribución de Valorcuota Personas que no pagan**

# In[229]:


Histog_Valorcuota_nopaga = plt.hist(Fintech_Balance[Fintech_Balance.Impago==1].Valorcuota.values, 10, facecolor="red", alpha=0.5)
plt.title("Distribución de Valorcuota que no pagan")


# - **Distribución de Valorcuota de personas que si pagan**

# In[230]:


Histog_Valorcuota_sipaga = plt.hist(Fintech_Balance[Fintech_Balance.Impago==0].Valorcuota.values, 10, facecolor="green", alpha=0.5)
plt.title("Distribución de Valorcuota que si pagan")


# - **Distribución de Valorhoy Personas que no pagan**

# In[231]:


#Histog_Valorhoy_nopaga = plt.hist(Fintech_Balance[Fintech_Balance.Impago==1].Valorhoy.values, 10, facecolor="red", alpha=0.5)
#plt.title("Distribución de Valorhoy que no pagan")


# - **Distribución de Valorhoy de personas que si pagan**

# In[232]:


#Histog_Valorhoy_sipaga = plt.hist(Fintech_Balance[Fintech_Balance.Impago==0].Valorhoy.values, 10, facecolor="green", alpha=0.5)
#plt.title("Distribución de Valorhoy que si pagan")


# - **¿Los Impagos dependen de la variable Sexo?:**

# In[233]:


plt.figure(figsize=(10,5))
dist_data = pandas.concat([Fintech_Balance[Fintech_Balance.Impago == 1].groupby("Sexo").Impago.count(),
                           Fintech_Balance[Fintech_Balance.Impago == 0].groupby("Sexo").Impago.count()], axis=1)
dist_data.columns = ["No_Pagan","Si_Pagan"]
dist_data_Final_N = dist_data.No_Pagan / (dist_data.No_Pagan + dist_data.Si_Pagan )
dist_data_Final_N=dist_data_Final_N.sort_values(ascending=False)
plt.figure(figsize=(10,4))
ax = dist_data_Final_N.plot(kind="bar", color="r", y="Percentage")


# In[234]:


plt.figure(figsize=(10,5))
dist_data = pandas.concat([Fintech_Balance[Fintech_Balance.Impago == 1].groupby("Sexo").Impago.count(),
                           Fintech_Balance[Fintech_Balance.Impago == 0].groupby("Sexo").Impago.count()], axis=1)
dist_data.columns = ["No_Pagan","Si_Pagan"]
dist_data_Final_N = dist_data.Si_Pagan / (dist_data.No_Pagan + dist_data.Si_Pagan )
dist_data_Final_N=dist_data_Final_N.sort_values(ascending=False)
plt.figure(figsize=(10,4))
ax = dist_data_Final_N.plot(kind="bar", color="g", y="Percentage")


# - ¿Los Impagos dependen de la variable **Ciudad**?:

# In[235]:


plt.figure(figsize=(10,5))
dist_data = pandas.concat([Fintech_Balance[Fintech_Balance.Impago == 1].groupby("Ciudad").Impago.count(),
                           Fintech_Balance[Fintech_Balance.Impago == 0].groupby("Ciudad").Impago.count()], axis=1)
dist_data.columns = ["No_Pagan","Si_Pagan"]
dist_data_Final_N = dist_data.No_Pagan / (dist_data.No_Pagan + dist_data.Si_Pagan )
dist_data_Final_N=dist_data_Final_N.sort_values(ascending=False)
plt.figure(figsize=(10,4))
ax = dist_data_Final_N.plot(kind="bar", color="r", y="Percentage")


# In[236]:


plt.figure(figsize=(10,5))
dist_data = pandas.concat([Fintech_Balance[Fintech_Balance.Impago == 1].groupby("Ciudad").Impago.count(),
                           Fintech_Balance[Fintech_Balance.Impago == 0].groupby("Ciudad").Impago.count()], axis=1)
dist_data.columns = ["No_Pagan","Si_Pagan"]
dist_data_Final_N = dist_data.Si_Pagan / (dist_data.No_Pagan + dist_data.Si_Pagan )
dist_data_Final_N=dist_data_Final_N.sort_values(ascending=False)
plt.figure(figsize=(10,4))
ax = dist_data_Final_N.plot(kind="bar", color="g", y="Percentage")


# - ¿Los Impagos dependen de la variable **Tipo_credito**?:

# In[237]:


plt.figure(figsize=(10,5))
dist_data = pandas.concat([Fintech_Balance[Fintech_Balance.Impago == 1].groupby("Tipo_credito").Impago.count(),
                           Fintech_Balance[Fintech_Balance.Impago == 0].groupby("Tipo_credito").Impago.count()], axis=1)
dist_data.columns = ["No_Pagan","Si_Pagan"]
dist_data_Final_N = dist_data.No_Pagan / (dist_data.No_Pagan + dist_data.Si_Pagan )
dist_data_Final_N=dist_data_Final_N.sort_values(ascending=False)
plt.figure(figsize=(10,4))
ax = dist_data_Final_N.plot(kind="bar", color="r", y="Percentage")


# In[238]:


plt.figure(figsize=(10,5))
dist_data = pandas.concat([Fintech_Balance[Fintech_Balance.Impago == 1].groupby("Tipo_credito").Impago.count(),
                           Fintech_Balance[Fintech_Balance.Impago == 0].groupby("Tipo_credito").Impago.count()], axis=1)
dist_data.columns = ["No_Pagan","Si_Pagan"]
dist_data_Final_N = dist_data.Si_Pagan / (dist_data.No_Pagan + dist_data.Si_Pagan )
dist_data_Final_N=dist_data_Final_N.sort_values(ascending=False)
plt.figure(figsize=(10,4))
ax = dist_data_Final_N.plot(kind="bar", color="g", y="Percentage")


# - ¿Los Impagos dependen de la variable **Calificación**?:

# In[239]:


#Fintech_Balance[Fintech_Balance.Impago == 1].groupby("Calificacion").Impago.count()
#Fintech_Balance[Fintech_Balance.Impago == 0].groupby("Calificacion").Impago.count()
#dist_data = pandas.concat([Fintech_Balance[Fintech_Balance.Impago == 1].groupby("Calificacion").Impago.count(),
 #                          Fintech_Balance[Fintech_Balance.Impago == 0].groupby("Calificacion").Impago.count()], axis=1)
#dist_data.columns = ["No_Pagan","Si_Pagan"]
#dist_data


# In[240]:


#Fintech_Balance.groupby('Calificacion').size()


# - ¿Los Impagos dependen de la variable **Actividad**?:

# In[241]:


Fintech_Balance[Fintech_Balance.Impago == 1].groupby("Actividad").Impago.count()
Fintech_Balance[Fintech_Balance.Impago == 0].groupby("Actividad").Impago.count()
dist_data = pandas.concat([Fintech_Balance[Fintech_Balance.Impago == 1].groupby("Actividad").Impago.count(),
                           Fintech_Balance[Fintech_Balance.Impago == 0].groupby("Actividad").Impago.count()], axis=1)
dist_data.columns = ["No_Pagan","Si_Pagan"]
dist_data


# In[242]:


Fintech_Balance.groupby('Actividad').size()


# In[243]:


plt.figure(figsize=(10,5))
dist_data = pandas.concat([Fintech_Balance[Fintech_Balance.Impago == 1].groupby("Actividad").Impago.count(),
                           Fintech_Balance[Fintech_Balance.Impago == 0].groupby("Actividad").Impago.count()], axis=1)
dist_data.columns = ["No_Pagan","Si_Pagan"]
dist_data_Final_N = dist_data.No_Pagan / (dist_data.No_Pagan + dist_data.Si_Pagan )
dist_data_Final_N=dist_data_Final_N.sort_values(ascending=False)
plt.figure(figsize=(10,4))
ax = dist_data_Final_N.plot(kind="bar", color="r", y="Percentage")


# In[244]:


plt.figure(figsize=(10,5))
dist_data = pandas.concat([Fintech_Balance[Fintech_Balance.Impago == 1].groupby("Actividad").Impago.count(),
                           Fintech_Balance[Fintech_Balance.Impago == 0].groupby("Actividad").Impago.count()], axis=1)
dist_data.columns = ["No_Pagan","Si_Pagan"]
dist_data_Final_N = dist_data.Si_Pagan / (dist_data.No_Pagan + dist_data.Si_Pagan )
dist_data_Final_N=dist_data_Final_N.sort_values(ascending=False)
plt.figure(figsize=(10,4))
ax = dist_data_Final_N.plot(kind="bar", color="g", y="Percentage")


# **Parte 9 : Arreglo de la matriz X.** 

# In[245]:


Fintech_Balance.shape


# - **Listado de Variables en la matriz X**

# In[246]:


Fintech_Balance.columns


# In[247]:


Fintech_Balance.head()
#238


# **Transformación de variables categoricas a Dummy de:** 
# - Sexo 
# - Ciudad   
# - Tipo_credito 
# - Calificacion 
# - Actividad

# In[248]:


### Creación de variables Dummy para Sexo:####
Sexo_dummies = pandas.get_dummies(Fintech_Balance.Sexo, prefix = "Sexo_")
# Se elimina la columna redundante
Sexo_dummies.drop(Sexo_dummies.columns[0], axis=1, inplace=True)
# Se concatena la data dummy con el data set original
Fintech_Balance_0 = pandas.concat([Fintech_Balance,Sexo_dummies], axis=1)

# Creación de variables Dummy para Ciudad:####
Ciudad_dummies = pandas.get_dummies(Fintech_Balance.Ciudad, prefix = "Ciudad_")
# Se elimina la columna redundante
Ciudad_dummies.drop(Ciudad_dummies.columns[0], axis=1, inplace=True)
# Se concatena la data dummy con el data set original
Fintech_Balance_1 = pandas.concat([Fintech_Balance_0,Ciudad_dummies], axis=1)

# Creación de variables Dummy para Tipo_credito:####
Tipo_credito_dummies = pandas.get_dummies(Fintech_Balance.Tipo_credito, prefix = "Tipo_credito_")
# Se elimina la columna redundante
Tipo_credito_dummies.drop(Tipo_credito_dummies.columns[0], axis=1, inplace=True)
# Se concatena la data dummy con el data set original
Fintech_Balance_2 = pandas.concat([Fintech_Balance_1,Tipo_credito_dummies], axis=1)

# Creación de variables Dummy para Calificacion:####
#Calificacion_dummies = pandas.get_dummies(Fintech_Balance.Calificacion, prefix = "Calificacion_")
# Se elimina la columna redundante
#Calificacion_dummies.drop(Calificacion_dummies.columns[0], axis=1, inplace=True)
# Se concatena la data dummy con el data set original
#Fintech_Balance_3 = pandas.concat([Fintech_Balance_2,Calificacion_dummies], axis=1)

# Creación de variables Dummy para Actividad:####
Actividad_dummies = pandas.get_dummies(Fintech_Balance.Actividad, prefix = "Actividad_")
# Se elimina la columna redundante
Actividad_dummies.drop(Actividad_dummies.columns[0], axis=1, inplace=True)
# Se concatena la data dummy con el data set original
Fintech_Balance_4 = pandas.concat([Fintech_Balance_2,Actividad_dummies], axis=1)


# In[249]:


Fintech_Balance_4.columns.values


# **Se listan las variables, para dejar solo las númericas y las dummy, eliminando las categoricas**
# - Sexo 
# - Ciudad   
# - Tipo_credito 
# - Calificacion 
# - Actividad

# - Se lista el tipo de variables numericas que tiene el data set

# In[250]:


Fintech_Balance_4.dtypes


# - Se dejan solo el data set con variables numericas.

# In[251]:


Feature=Fintech_Balance_4.select_dtypes(['float64','uint8',"int64"])
del Feature["Impago"]
#del Feature["Valorcuota"]


# In[252]:


Feature
Feature.to_excel (r'antesfinal.xlsx', index = False, header = True)
Feature.columns.values
Var_Entrena = Feature.columns.values


# **Creación de la matriz de las variables independientes, se llama X (No deben de aparecer caracteres)**

# In[253]:


Feature


# In[254]:


X = Feature


# **Declaración del vector, llamada variable dependiente y**
# - Impago van a ser mayores o iguales a 30 días.

# In[255]:


y = Fintech_Balance_4["Impago"].values


# **Se Realiza la partición de los datos**

# In[256]:


#  Defino la variable target y Particionamiento de los datos.
# 50% Entrenamiento. 50% validación.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state = 100)  


#  **Paso 11 : Árbol de decisión.**

# In[257]:


clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth = 6, min_samples_leaf= 15 )
clf_gini.fit(X_train, y_train)


# In[258]:


feat_importance=pandas.DataFrame({"feature":Feature, "importance":clf_gini.feature_importances_})
#clf_gini.feature_importances_


# **Se Calculan las variables más importantes del modelo.**

# In[259]:


feat_importance.sort_values("importance", ascending=False).head()


# In[260]:


feat_importance.to_csv("features_importance.csv")


# **Se Realiza la predicción de "y" en función de la data de pruebas**
# 
# La variable resultante va a ser **y_predict** y va a ser en función de la data X_test, van a ser valores dicotomicos 0 ó 1.Tambien se va a calcular **y_predictions_Proba** que son las predicciones en función de la data X_test, en terminos de probabilidad de que suceda el evento (Impago) y su complemento.
# 
# Se realiza una tranformación de matriz a dataframe de la variable **y_predictions_Proba** y el dataframe se llamará: **y_predictions_Proba_1** con las columnas **Calificación** y **Y_test_Prob_1**.

# In[261]:


y_predict = clf_gini.predict(X_test)
y_predictions_Proba = clf_gini.predict_proba(X_test)
y_predictions_Proba_1 = pandas.DataFrame(data=y_predictions_Proba, columns=["Y_test_Prob_0","Y_test_Prob_1"])
y_predictions_Proba_1

bins = [-0.000000001, 0.200001, 0.400001, 0.600001, 0.7000001, 0.85000001, 1.000000001]
names = ["C","CC","B","BB","A","AA"]
y_predictions_Proba_1["Y_test_Prob_0"]=pandas.cut(y_predictions_Proba_1["Y_test_Prob_0"],bins, labels=names)
y_predictions_Proba_1[0:8]
#asas=y_predictions_Proba_1["Y_test_Prob_1"]-1
y_predictions_Proba_1.iloc[0:8]


# **Categorización de las variables en CC, C, B, BB, A, AA**
# 
# Se asignan los cortes, según la distribución de SER BUEN PAGADOR.
# 
# Por ejemplo entre: 
# 
# - Mayor 85 se asignará calificación de impago de **AA**.
# 
# - 70-85: Se le asignará una calificación de **A**.
# 
# - 50-70: Se le aginará una calificación de **BB**.
# 
# - 40-50: Se le aginará una calificación de **B**.
# 
# - 20-40: Se le aginará una calificación de **CC**.
# 
# - 0-20: Se le aginará una calificación de **C**.

# **Áca se realiza la comprobación de la distribución de las probabilidades de impagos, para así asignar las variables categoricas.**

# In[262]:


Histog_Pronos_Impago = plt.hist(y_predictions_Proba_1.Y_test_Prob_1.values, 6, facecolor="red", alpha=0.5)
plt.title("Distribución de Pronosticos Impago")


# Se pasa de vectores **y_test** y **y_predict** a DataFrame Y_test_Real y Y_test_pron, respectivamente

# In[263]:


Y_test_Real = pandas.DataFrame(data=y_test, columns=["Y_test_Real"])
Y_test_pron = pandas.DataFrame(data=y_predict, columns=["Y_test_Pron"])
Comparativo = pandas.concat([Y_test_Real,Y_test_pron], axis=1)
Comparativo.iloc[0:7]


# Se concatena, la data de test + pronosticos dicotomicos + las probabilidades.

# In[264]:


#  Anexo la calificación a los datos de validación. 
Fintech_train, Fintech_test = train_test_split(Fintech_Balance, test_size=0.5, random_state = 100)
Fintech_test.reset_index(drop=True, inplace=True)
Fintech_test
Fintech_Final = pandas.concat([Fintech_test,Y_test_pron,y_predictions_Proba_1], axis=1)
Fintech_Final.iloc[0:8]


# Exactitud del Modelo

# In[265]:


accuracy_score(y_test , y_predict) * 100


# Matriz de confusión

# In[266]:


pandas.crosstab(y_test ,y_predict
               ,rownames = ["Real"]
               ,colnames = ["Pronostico"])


# Resumen de los estadisticos de evaluacion del modelo.

# In[267]:


print("\n \n")
print(classification_report(y_test, y_predict))


# **Modelo de Regresion Logistica.**

# In[268]:


Regession_log = LogisticRegression()
Regession_log.fit(X_train,y_train)


# In[269]:


y_Pred = Regession_log.predict(X_test)
y_Pred 

#Matriz = confusion_matrix(y_test, y_Pred)
#print(Matriz)

Precision_log = precision_score(y_test, y_Pred)
print(Precision_log)


# - Matriz de confusión

# In[270]:


pandas.crosstab(y_test ,y_Pred
               ,rownames = ["Real"]
               ,colnames = ["Pronostico"])


# - Listado de coeficientes del modelo.

# In[271]:


List_Coefi=Regession_log.coef_
List_Coefi
List_Coefi.transpose()
Coeficientes= pandas.DataFrame(data=List_Coefi.transpose(),columns=["Coeficientes"])
Coeficientes.to_excel (r'Coeficientes_Regression_Finte.xlsx', index = False, header = True)


# **Modelo de Random Forest.**

# In[272]:


from sklearn.ensemble import RandomForestClassifier
RandomForest = RandomForestClassifier(max_depth = 6, min_samples_leaf= 15)

RandomForest.fit(X_train,y_train)

pred_train = RandomForest.predict(X_train)
pred_test =  RandomForest.predict(X_test)

#Matriz = confusion_matrix(y_test, pred_test)
#print(Matriz)
accuracy_score(y_test , y_predict) * 100
from sklearn.metrics import accuracy_score
accuracy_train = accuracy_score(pred_train,y_train)
accuracy_test = accuracy_score(pred_test,y_test)
roc_auc_score(y_test , pred_test)


# - Matriz de confusión

# In[273]:


pandas.crosstab(y_test ,pred_test
               ,rownames = ["Real"]
               ,colnames = ["Pronostico"])


# **Modelo de Red Nuronal**
# - imput_dim tiene que se la dimension de la matriz X.

# In[274]:


#conda install -- keras. pip install tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense
clasificador = Sequential()
# Se realiza un conteo de culumnas:
X_Dim = pandas.DataFrame(data=X)
# Creamos la capa oculta y la primera capa
clasificador.add(Dense(units=6, kernel_initializer="uniform", activation="relu",input_dim=X_Dim.shape[1]))
# Creamos la segunda capa oculta
clasificador.add(Dense(units=6, kernel_initializer="uniform", activation="relu"))
# Creamos la capa de salida
clasificador.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))
# Descenso de gradiente estocastico que es compilar el modelo
clasificador.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
# Ajustando la red neuroral al data set
clasificador.fit(X_train, y_train, batch_size=10, epochs= 100)


# In[275]:


y_test_red =   clasificador.predict(X_test)
y_train_red =  clasificador.predict(X_train)


# - Matriz de confusión

# **Indices de Contraste:**

# - **Índice ROC Árbol Decisión:**
# 
# Como el cálculo es financiero su pronóstico va a ser en terminos de puntajes, por ende la estadística de validación, va a ser el índice ROC. 

# In[276]:


# Valor ROC
roc_auc_score(y_test , y_predict)*100


# - **Índice ROC Regresión Logistica:**

# In[277]:


# Valor ROC
roc_auc_score(y_test , y_Pred)*100


# - **Índice ROC Regresión Random Forest:**

# In[278]:


# Valor ROC
roc_auc_score(y_test , pred_test)*100


# - **Índice ROC Red Neuronal**

# In[279]:


# Valor ROC
roc_auc_score(y_test , y_test_red)*100


# **Cálculo de las curva ROC**

# In[280]:


# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test , y_predict) # Arbol
lr_fpr, lr_tpr, _ = roc_curve(y_test , y_Pred) # Logistica
rf_fpr, rf_tpr, _ = roc_curve(y_test , pred_test) # Random Forest
rn_fpr, rn_tpr, _ = roc_curve(y_test , pred_test) # red neuronal
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Arbol')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistica')
plt.plot(rf_fpr, rf_tpr, marker='+', label='Random Forest')
#plt.plot(y_test , pred_test, marker='*', label='red neuronal')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()


# **Exportación del Resultado, del data set de pruebas del árbol de decisión :**
# 
# Se exporta a un excel los resultados, su ubicacion es la carpeta de inicio de **AWS.** 

# In[281]:


#Regresion Logistica
y_Predict_RL = Regession_log.predict(X_test)
y_predictions_Proba_RL = Regession_log.predict_proba(X_test)
y_predictions_Proba_RL = pandas.DataFrame(data=y_predictions_Proba_RL, columns=["RL_Y_test_Prob_0","RL_Y_test_Prob_1"])
y_predictions_Proba_RL["RL_Y_test_Prob_0"]=pandas.cut(y_predictions_Proba_RL["RL_Y_test_Prob_0"],bins, labels=names)
y_predictions_Proba_RL[0:8]
Fintech_Final = pandas.concat([Fintech_test,y_predictions_Proba_1, y_predictions_Proba_RL], axis=1)
#C_Coop_Final

#Random Forest
y_Predict_RF = RandomForest.predict(X_test)
y_predictions_Proba_RF = RandomForest.predict_proba(X_test)
y_predictions_Proba_RF = pandas.DataFrame(data=y_predictions_Proba_RF, columns=["RF_Y_test_Prob_0","RF_Y_test_Prob_1"])
y_predictions_Proba_RF["RF_Y_test_Prob_0"]=pandas.cut(y_predictions_Proba_RF["RF_Y_test_Prob_0"],bins, labels=names)
y_predictions_Proba_RF[0:8]
Fintech_Final = pandas.concat([Fintech_test,y_predictions_Proba_1, y_predictions_Proba_RL,y_predictions_Proba_RF], axis=1)
Fintech_Final


# In[282]:


# Exportando a Excel data set calificada.
Fintech_Final.to_excel (r'F_BDFintech.xlsx', index = False, header = True)


# In[283]:


# Exportando data set Balanceados.
#Fintech_Balance.to_excel (r'F_BDFintech.xlsx', index = False, header = True)


# **Paso 11 : Implementación.**
# 
# **Guardar modelo para uso futuro**
# 
# Finalmente, ya desarrollado el modelo y evaluado con todas las métricas ahora se esta listo para implementar el modelo en producción. El último paso antes de la implementación es guardar nuestro modelo, utilizando el código a continuación.

# **Se Corre este Programa con la data que se quiere calificar.**

# In[284]:


#import mpld3
# Import excel
anno="2020"
mes = "04"
dia = "21"
C_Fintechdiamesanno = pandas.read_excel(f'C_BDFintech{dia}{mes}{anno}.xlsx', sheet_name='Sheet1')
#Renombrando el nombre de las variables, eliminando espacios y caracteres especiales.
# Data Fintech
C_Fintechdiamesanno1 = Fintechdiamesanno.rename(columns= {"N° Credito":"NCredito",
                                                    "Monto Credito":"MontoCredito",
                                                    "Fecha De Pago":"FechaDePago",
                                                     "Fecha Desembolso":"FechaDesembolso",   
                                                    "Valor cuota":"Valorcuota",
                                                    "Valor hoy":"Valorhoy",
                                                    "Capital vencido":"Capitalvencido",
                                                    "Interes vencido":"Interesvencido",
                                                    "0-30 dias":"_0_30_dias",
                                                    "31-60 dias":"_31_60_dias",
                                                    "61-90 dias":"_61_90_dias",
                                                    "91-120 dias":"_91_120_dias",
                                                    "121-150 dias":"_121_150_dias",
                                                    "151-180 dias":"_151_180_dias",
                                                    "Mayor 180 dias":"_Mayor_180_dias",
                                                    "Tipo credito":"Tipo_credito",
                                                    "Calificación":"Calificacion",
                                                    "Dias de mora":"Dias_de_mora",
                                                    "Num.Cuotas Vencidas":"Num_Cuotas_Vencidas",
                                                    "Fecha.Nac":"Fecha_Nac",
                                                    "Capita NO Venc.":"Capita_NO_Venc",
                                                     "Total vencido":"Totalvencido"})
C_Fintechdiamesanno1.iloc[0:3]


# **Paso 6 : Eliminación de variables Irrelevantes para el pronóstico.**
# 
# - En este paso se procede a eliminar las variables que lo aportan a la predicción.
# - Mas de 100 categorias.

# In[285]:


#Coopdiamesanno1.drop(["F_INICIOFI","F_CARGOPRE",'SCORE',"ANNO", "SALDO_CAP_2020_1","SALDO_PROVI_2020_1"],axis=1)
del C_Fintechdiamesanno1["Nombre"] # + 100 categorias
del C_Fintechdiamesanno1["NCredito"] # + 100 categorias
del C_Fintechdiamesanno1["sucursal"] # es una costante = 1.
del C_Fintechdiamesanno1["FechaDePago"] # Fecha de pago, es una variable que ocurre a partir del evento.
del C_Fintechdiamesanno1["FechaDesembolso"] # Fecha de desembolso es una variable que ocurre a partir del evento.
del C_Fintechdiamesanno1["Mora"] # Es una constante =0
del C_Fintechdiamesanno1["Totalvencido"]  # es una variable que se da a partir del evento 
del C_Fintechdiamesanno1["_0_30_dias"] # es una variable que se da a partir del evento
del C_Fintechdiamesanno1["_31_60_dias"] # es una variable que se da a partir del evento
del C_Fintechdiamesanno1["_61_90_dias"] # es una variable que se da a partir del evento
del C_Fintechdiamesanno1["_91_120_dias"] # es una variable que se da a partir del evento
del C_Fintechdiamesanno1["_121_150_dias"] # es una variable que se da a partir del evento
del C_Fintechdiamesanno1["_151_180_dias"] # es una variable que se da a partir del evento
del C_Fintechdiamesanno1["_Mayor_180_dias"] # es una variable que se da a partir del evento.
del C_Fintechdiamesanno1["Dias_de_mora"] # es una variable que se da a partir del evento.
del C_Fintechdiamesanno1["Capita_NO_Venc"] # es una variable que se da a partir del evento.
del C_Fintechdiamesanno1["Interesvencido"] # Con el campo edad se representa esta variable.
del C_Fintechdiamesanno1["Capitalvencido"] # Con el campo edad se representa esta variable.
del C_Fintechdiamesanno1["Fecha_Nac"] # Con el campo edad se representa esta variable.
del C_Fintechdiamesanno1["Num_Cuotas_Vencidas"] # Con el campo edad se representa esta variable.
del C_Fintechdiamesanno1["Calificacion"] # Con el campo edad se representa esta variable.
del C_Fintechdiamesanno1["Valorhoy"] #esta variable toma relevancia cuando hay impagos
del C_Fintechdiamesanno1["Saldo"] # Este campo se origina despues de otorgar el crédito.


# **Parte 6 : Limpieza del Data Set.**

# - En la variable **Ingresos** se cambia los ceros por valores vacios.
# 

# In[286]:


#Limpieza de los valores.
# Se reemplazan los valores ceros por vacios
C_Fintechdiamesanno1[['score', "Ingresos"]] = C_Fintechdiamesanno1[['score', "Ingresos"]].replace(0,np.NaN)
#Imputación MontoCredito
C_Fintechdiamesanno1["MontoCredito"] = C_Fintechdiamesanno1["MontoCredito"].replace(np.NaN,C_Fintechdiamesanno1["MontoCredito"].median())
#C_Fintechdiamesanno1["Valorhoy"] =     C_Fintechdiamesanno1["Valorhoy"].replace(np.NaN,0)
C_Fintechdiamesanno1["Ingresos"] =     C_Fintechdiamesanno1["Ingresos"].replace(np.NaN,C_Fintechdiamesanno1["Ingresos"].median())
C_Fintechdiamesanno1["score"] =        C_Fintechdiamesanno1["score"].replace(np.NaN,C_Fintechdiamesanno1["score"].median())
C_Fintechdiamesanno1["Actividad"] =    C_Fintechdiamesanno1["Actividad"].replace(np.NaN,"INDEPENDIENTE")


# In[287]:


C_Fintech_Balance=C_Fintechdiamesanno1


# **Transformación de variables categoricas a Dummy de:** 
# - Sexo 
# - Ciudad   
# - Tipo_credito 
# - Calificacion 
# - Actividad

# In[288]:


### Creación de variables Dummy para Sexo:####
Sexo_dummies = pandas.get_dummies(C_Fintech_Balance.Sexo, prefix = "Sexo_")
# Se elimina la columna redundante
Sexo_dummies.drop(Sexo_dummies.columns[0], axis=1, inplace=True)
# Se concatena la data dummy con el data set original
C_Fintech_Balance_0 = pandas.concat([C_Fintech_Balance,Sexo_dummies], axis=1)

# Creación de variables Dummy para Ciudad:####
Ciudad_dummies = pandas.get_dummies(C_Fintech_Balance.Ciudad, prefix = "Ciudad_")
# Se elimina la columna redundante
Ciudad_dummies.drop(Ciudad_dummies.columns[0], axis=1, inplace=True)
# Se concatena la data dummy con el data set original
C_Fintech_Balance_1 = pandas.concat([C_Fintech_Balance_0,Ciudad_dummies], axis=1)

# Creación de variables Dummy para Tipo_credito:####
Tipo_credito_dummies = pandas.get_dummies(C_Fintech_Balance.Tipo_credito, prefix = "Tipo_credito_")
# Se elimina la columna redundante
Tipo_credito_dummies.drop(Tipo_credito_dummies.columns[0], axis=1, inplace=True)
# Se concatena la data dummy con el data set original
C_Fintech_Balance_2 = pandas.concat([C_Fintech_Balance_1,Tipo_credito_dummies], axis=1)

# Creación de variables Dummy para Calificacion:####
#Calificacion_dummies = pandas.get_dummies(C_Fintech_Balance.Calificacion, prefix = "Calificacion_")
# Se elimina la columna redundante
#Calificacion_dummies.drop(Calificacion_dummies.columns[0], axis=1, inplace=True)
# Se concatena la data dummy con el data set original
#C_Fintech_Balance_3 = pandas.concat([C_Fintech_Balance_2,Calificacion_dummies], axis=1)

# Creación de variables Dummy para Actividad:####
Actividad_dummies = pandas.get_dummies(C_Fintech_Balance.Actividad, prefix = "Actividad_")
# Se elimina la columna redundante
Actividad_dummies.drop(Actividad_dummies.columns[0], axis=1, inplace=True)
# Se concatena la data dummy con el data set original
C_Fintech_Balance_4 = pandas.concat([C_Fintech_Balance_2,Actividad_dummies], axis=1)


# **Se dejan solo el data set con variables numericas.**
# 
# - En este paso se eliguen solo las columnas del entrenamiento (Queda automatizado).

# In[289]:


#De las matriz de variables Dummy, se dejan solo las variables numericas 
C_features=C_Fintech_Balance_4.select_dtypes(['float64','uint8',"int64"])
C_X=C_features[Var_Entrena].values


# In[290]:


C_X


# **Se Califica la data usando los diferentes modelos.**

# In[291]:


# Árbol de decisión.
C_y_predict = clf_gini.predict(C_X)
C_y_predict
C_y_predictions_Proba = clf_gini.predict_proba(C_X)
C_y_predictions_Proba_1 = pandas.DataFrame(data=C_y_predictions_Proba, columns=["C_Arbol_Y_test_Prob_0","C_Arbol_Y_test_Prob_1"])
C_y_predictions_Proba_1

bins = [-0.000000001, 0.200001, 0.400001, 0.600001, 0.7000001, 0.85000001, 1.000000001]
names = ["C","CC","B","BB","A","AA"]
C_y_predictions_Proba_1["C_Arbol_Y_test_Prob_0"]=pandas.cut(C_y_predictions_Proba_1["C_Arbol_Y_test_Prob_0"],bins, labels=names)
C_y_predictions_Proba_1[0:8]
#asas=y_predictions_Proba_1["C_Y_test_Prob_1"]-1
C_y_predictions_Proba_1.iloc[0:8]  

#C_Coop_Final = pandas.concat([C_Coopdiamesanno,C_y_predictions_Proba_1], axis=1)
#C_Coop_Final
#C_Coop_Final.iloc[0:8]

#Regresion Logistica
C_y_Predict_RL = Regession_log.predict(C_X)
C_y_predictions_Proba_RL = Regession_log.predict_proba(C_X)
C_y_predictions_Proba_RL = pandas.DataFrame(data=C_y_predictions_Proba_RL, columns=["C_RL_Y_test_Prob_0","C_RL_Y_test_Prob_1"])
C_y_predictions_Proba_RL["C_RL_Y_test_Prob_0"]=pandas.cut(C_y_predictions_Proba_RL["C_RL_Y_test_Prob_0"],bins, labels=names)
C_y_predictions_Proba_RL[0:8]
#C_Coop_Final = pandas.concat([C_Coopdiamesanno,C_y_predictions_Proba_1, C_y_predictions_Proba_RL], axis=1)
#C_Coop_Final

#Random Forest
C_y_Predict_RF = RandomForest.predict(C_X)
C_y_predictions_Proba_RF = RandomForest.predict_proba(C_X)
C_y_predictions_Proba_RF = pandas.DataFrame(data=C_y_predictions_Proba_RF, columns=["C_RF_Y_test_Prob_0","C_RF_Y_test_Prob_1"])
C_y_predictions_Proba_RF["C_RF_Y_test_Prob_0"]=pandas.cut(C_y_predictions_Proba_RF["C_RF_Y_test_Prob_0"],bins, labels=names)
C_y_predictions_Proba_RF[0:8]
C_Fintech_Final = pandas.concat([C_Fintechdiamesanno,C_y_predictions_Proba_1, C_y_predictions_Proba_RL,C_y_predictions_Proba_RF], axis=1)
C_Fintech_Final

#Red Neuronal
C_y_Predict_RN = clasificador.predict(C_X)
C_y_predictions_Proba_RN = clasificador.predict_proba(C_X)
#C_y_predictions_Proba_RN = pandas.DataFrame(data=C_y_predictions_Proba_RN, columns=["C_RF_Y_test_Prob_0","C_RN_Y_test_Prob_1"])
pandas.C_RN_Y_test_Prob_0 = C_y_predictions_Proba_RN - 1
pandas.C_RN_Y_test_Prob_0
#print(C_RN_Y_test_Prob_1)
#C_y_predictions_Proba_RN=pandas.concat([C_RN_Y_test_Prob_0,C_RN_Y_test_Prob_1], axis=1)
#C_y_predictions_Proba_RN = pandas.DataFrame(data=C_y_predictions_Proba_RN, columns=["C_RN_Y_test_Prob_0","C_RN_Y_test_Prob_1"])
#C_y_predictions_Proba_RN["C_RN_Y_test_Prob_0"]=pandas.cut(C_y_predictions_Proba_RN["C_RN_Y_test_Prob_0"],bins, labels=names)
#C_y_predictions_Proba_RN[0:8]
#C_Coop_Final = pandas.concat([C_Coopdiamesanno,C_y_predictions_Proba_1, C_y_predictions_Proba_RL,C_y_predictions_Proba_RF,C_y_predictions_Proba_RN], axis=1)
C_Fintech_Final


# In[292]:


# Exportando a Excel data set calificada.
C_Fintech_Final.to_excel (r'C_F_BDFintech.xlsx', index = False, header = True)


# **Puntuación de datos nuevos**
# 
# Para la puntuación, necesitamos cargar nuestro objeto modelo (clf) y el objeto codificador de etiquetas nuevamente en el entorno python.

# Se carga el nuestro nuevo conjunto de datos y lo pasamos a la macro de puntuación.
