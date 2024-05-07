import pandas as pd
import matplotlib. pyplot as plt
import numpy as np

data = pd.read_csv('data_C02_emission.csv')


plt.figure()
data['CO2 Emissions (g/km)'].plot(kind = 'hist')
plt.show()


data['Make']=pd.Categorical(data['Make'])
data["Model"] = pd.Categorical(data["Model"])
data['Vehicle Class']=pd.Categorical(data['Vehicle Class'])
data['Transmission']=pd.Categorical(data['Transmission'])
data['Fuel Type']=pd.Categorical(data['Fuel Type'])
data.plot.scatter(x = 'Fuel Consumption City (L/100km)', y = 'CO2 Emissions (g/km)', c = 'Fuel Type')
plt.show()


data.boxplot(column = ['Fuel Consumption Hwy (L/100km)'], by = 'Fuel Type')
plt.show()


data.groupby('Fuel Type').size().plot(kind = 'bar')
plt.show()


data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean().plot(kind = 'bar')
plt.show()