#from urllib import request
#url='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/china_gdp.csv'
#request.urlretrieve(url,'china_gdp.csv')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('china_gdp.csv')
df.columns
df.head()

#check the data
#plt.scatter(df[['Year']],df[['Value']])

#plt.ylabel('GDP')
#plt.xlabel('Year')
#plt.savefig('gdp-year.png',dpi=300)
#plt.show()

#gdp looks like sigmoid function

#checking what kind of function will fit

def sigmoid(x,b1,b2):
	y= (1/(1 + np.exp((-b1)*(x-b2))))
	return y


###data with trial sigmoid model

#plt.scatter(df[['Year']],df[['Value']],color='blue')
#plt.plot(df[['Year']].values,sigmoid(df[['Year']].values,0.1,1990)*1.5*10**13,color='red',label='trial model') #choosing random values of b1,b2 and multiplying random number to get close to our model. 
#plt.ylabel('GDP')
#plt.xlabel('Year')

#plt.legend()
#plt.savefig('gdp-year-value-trail-model.png',dpi=300)
#plt.show()


#we need to find b1 and b2 such that the fit gets better. A way is to  minimise squared residuals of sigmoid(xdata, *popt) - ydata

from scipy.optimize import curve_fit
x_data=df[['Year']].values # or np.asanyarray(df[['Year']])
y_data=df[['Value']].values

x_norm=x_data/max(x_data)
y_norm=y_data/max(y_data)

#since it is sigmoid function, which goes from -1 to 1, we normalise the values

x_norm=x_norm.flatten()
y_norm=y_norm.flatten()
popt, pcov = curve_fit(sigmoid, x_norm, y_norm)
print('b1=',popt[0])
print('b2=',popt[1])

plt.scatter(x_norm,y_norm,color='blue')
plt.plot(x_norm,sigmoid(x_norm,690.4517142174786,0.9972071272537256),color='red',label='model')
plt.xlabel('Year-normalised')
plt.ylabel('GDP-normalised')
plt.legend()
plt.savefig('gdp-year-value-model.png',dpi=300)
#plt.show()


#checking for the accuracy of this dataset

msk=np.random.rand(len(df)) <0.8
train_x=x_norm[msk]
train_y=y_norm[msk]
test_x=x_norm[~msk]
test_y=y_norm[~msk]

popt, pcov = curve_fit(sigmoid,train_x,train_y)
pred_y = sigmoid(test_x,*popt)
from sklearn.metrics import r2_score
print("R2-score: %.2f" % r2_score(test_y,pred_y) )










