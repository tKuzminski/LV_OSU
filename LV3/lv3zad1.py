import pandas as pd
import numpy as np

data = pd.read_csv('data_C02_emission.csv')

print(len(data))
print(data.info())
print(data.isnull().sum(), data.duplicated().sum())
data = data.dropna().drop_duplicates().reset_index()
data['Make'] = pd.Categorical(data.Make)
data['Model'] = pd.Categorical(data.Model)
data['Vehicle Class'] = pd.Categorical(data['Vehicle Class'])
data['Transmission'] = pd.Categorical(data.Transmission)
data['Fuel Type'] = pd.Categorical(data['Fuel Type'])


data_sort_fuel_cons=data.sort_values(by = 'Fuel Consumption City (L/100km)')
print(data_sort_fuel_cons.tail(3)[['Make', 'Model', 'Fuel Consumption City (L/100km)']])
print(data_sort_fuel_cons.head(3)[['Make', 'Model', 'Fuel Consumption City (L/100km)']])


data_filt_by_eng=data [(data['Engine Size (L)'] > 2.5 ) & (data['Engine Size (L)'] < 3.5)]
print(len(data_filt_by_eng))
print(data_filt_by_eng['CO2 Emissions (g/km)'].mean())


data_audi = data[data['Make'] == 'Audi']
print(len(data_audi))
print(data_audi[data_audi['Cylinders'] == 4]['CO2 Emissions (g/km)'].mean())


print(data[(data['Cylinders'] >= 4) & (data['Cylinders'] % 2 == 0)]['index'].count())
data_cylinders = data.groupby('Cylinders')
print(data_cylinders[['CO2 Emissions (g/km)']].mean())


print(data[data['Fuel Type'] == 'D']['Fuel Consumption City (L/100km)'].mean())
print(data[data['Fuel Type'] == 'X']['Fuel Consumption City (L/100km)'].mean())
print(data[data['Fuel Type'] == 'D']['Fuel Consumption City (L/100km)'].median())
print(data[data['Fuel Type'] == 'X']['Fuel Consumption City (L/100km)'].median())


data_4cyl_diesel_ = data[(data['Cylinders'] == 4) & (data['Fuel Type'] == 'D')]
print(data_4cyl_diesel_.sort_values(by = 'Fuel Consumption City (L/100km)').tail(1)[['Make', 'Model']])


print(data[data['Transmission'].str.startswith('M')]['index'].count())


print(data.corr(numeric_only = True))

# Ako je korelacija blizu 1, to ukazuje na snažnu pozitivnu korelaciju, što znači da se vrijednosti dviju varijabli kreću u istom smjeru. 
# Ako je korelacija blizu -1, to ukazuje na snažnu negativnu korelaciju, što znači da se vrijednosti dviju varijabli kreću u suprotnim smjerovima. 
# Ako je korelacija blizu 0, to ukazuje na slabu ili nikakvu korelaciju između varijabli, varijable se ne mijenjaju u istom smjeru niti suprotno jedna drugoj.