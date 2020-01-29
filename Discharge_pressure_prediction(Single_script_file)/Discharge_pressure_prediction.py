import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import LinearRegression

def data_load_first(the_file):
    train = pd.read_csv(the_file,sep=r'\s*,\s*',header=0, encoding='ascii', engine='python')
    train_original=train.copy()
    return train,train_original

def seeing_column_name_first(train):
    for col in train.columns: 
        print(col) 
    print("\n")

def seeing_empty_value_first(train):
    null_columns=train.columns[train.isnull().any()]
    print("\n")
    print(train[train["Temperature"].isnull()][null_columns])
    print("\n")
    return train

def clean_nothing_dataset_first(train):
    print(train.isnull().sum())
    print("\n")
    train = train.dropna()
    print(train.isnull().sum())
    print("\n")
    return train

def removing_column_first(train):
    to_drop = ['Timestamp','Suction Pressure (psig)','Suction Temperature (F)','Speed (rpm)','By-pass Valve Position (%)','Discharge Temperature (F)','Run Status']
    train.drop(to_drop,inplace=True,axis=1)
    return train

def renaming_column_first(train):
    new_name = {'Total Flow (gpm)':'Total_Flow_(gpm)','Discharge Pressure (psig)':'Discharge_Pressure_(psig)'}
    train.rename(columns = new_name,inplace = True)
    print(train.head(5))
    print("\n")
    return train

def save_modified(train,op_file):
    train.to_csv(op_file)
    return train

def Discharge_Pressure_linear(Total_Flow):
    result = lin.predict(np.array(Total_Flow).reshape(1, -1))
    print(result[0,].shape) 
    return(result[0,])

def Discharge_Pressure_polynomial(Total_Flow): 
    Total_Flow = np.array(Total_Flow).reshape(1, -1)
    Total_Flow = poly.fit_transform(Total_Flow)
    result = lin2.predict(np.array(Total_Flow).reshape(1, -1))
    return(result[0,])


if __name__ == '__main__':

    # Data cleaning and prepration
    train_0,train_original_0 = data_load_first("Expander_data.csv")
    seeing_column_name_first(train_0)
    train_1 = removing_column_first(train_0)
    train_2 = renaming_column_first(train_1)
    train_3 = save_modified(train_2,'GPM_PSIG.csv')
    # Regression 
    new_df = pd.read_csv('GPM_PSIG.csv') 
    X = new_df.iloc[:, 1:2].values 
    y = new_df.iloc[:, 2].values 
    plt.scatter(X,y,color="red")
    plt.title('Total_Flow_(gpm) vs Discharge_Pressure_(psig)')
    plt.xlabel('Total_Flow_(gpm)')
    plt.ylabel('Discharge_Pressure_(psig)')
    plt.show()
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state=15)


    # Fitting Linear Regression to the dataset 
    lin = LinearRegression() 
    lin.fit(X_train, y_train) 
    # Fitting Polynomial Regression to the dataset 
    poly = PolynomialFeatures(degree = 3) 
    X_poly = poly.fit_transform(X_train) 
    poly.fit(X_poly, y_train) 
    lin2 = LinearRegression() 
    lin2.fit(X_poly, y_train) 
    # Visualising the Linear Regression results 
    plt.scatter(X_test, y_test, color = 'blue') 
    plt.plot(X_train, lin.predict(X_train), color = 'red') 
    plt.title('Linear Regression') 
    plt.xlabel('Total_Flow_(gpm)') 
    plt.ylabel('Discharge_Pressure_(psig)') 
    plt.show()
    # Visualising the Polynomial Regression results 
    plt.scatter(X_test, y_test, color = 'blue') 
    plt.plot(X_train, lin2.predict(X_poly), color = 'red') 
    # Avioding overflow error in matplotlib
    plt.rcParams['agg.path.chunksize'] = 1000
    plt.title('Polynomial Regression') 
    plt.xlabel('Total_Flow_(gpm)') 
    plt.ylabel('Discharge_Pressure_(psig)') 
    plt.show() 
    plt.scatter(X_train,y_train,color="blue") 
    plt.plot(X_train, lin.predict(X_train),color="red",linewidth=3) 
    plt.title('Linear Regression(training Set)')
    plt.xlabel('Total_Flow_(gpm)')
    plt.ylabel('Discharge_Pressure_(psig)')
    plt.show()
    plt.scatter(X_train,y_train,color="blue")  
    plt.plot(X_train, lin2.predict(X_poly),color="red",linewidth=3)
    plt.title('Polynomial Regression(training Set)')
    plt.xlabel('Total_Flow_(gpm)')
    plt.ylabel('Discharge_Pressure_(psig)')
    plt.show()
    from sklearn.metrics import r2_score,mean_squared_error
    y_pred = lin.predict(X_test)
    print(X_test.shape)
    print("###### For Linear Regression ######")
    print('R2 score: %.2f' % r2_score(y_test,y_pred)) 
    print('Mean squared Error :',mean_squared_error(y_test,y_pred))
    print("\n")
    y_pred = lin2.predict(poly.fit_transform(X_test))
    print("###### For Polynomial Regression ######")
    print('R2 score: %.2f' % r2_score(y_test,y_pred)) 
    print('Mean squared Error :',mean_squared_error(y_test,y_pred)) 
    print("\n")
    print("###### For Linear Regression ######")
    Total_Flow = float(input('Enter the Total_Flow_(gpm) : ')) 
    print('This Discharge_Pressure will be : ',float(Discharge_Pressure_linear(Total_Flow)),'psig')
    print("\n")
    print("###### For Polynomial Regression ######")
    Total_Flow = float(input('Enter the Total_Flow_(gpm) : ')) 
    print('This Discharge_Pressure will be : ',float(Discharge_Pressure_polynomial(Total_Flow)),'psig')
    print("\n")
