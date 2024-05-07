import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.linear_model as lm
from sklearn.metrics import max_error
from sklearn.preprocessing import OneHotEncoder


data = pd.read_csv('data_C02_emission.csv')

ohe = OneHotEncoder()
fuel_type_encoded = ohe.fit_transform(data[['Fuel Type']]).toarray()
data[ohe.categories_[0]] = fuel_type_encoded

y = data['CO2 Emissions (g/km)'].copy()
X = data.drop('CO2 Emissions (g/km)', axis = 1)
X_train_all, X_test_all, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

input_variable = ['Engine Size (L)', 'Cylinders', 'Fuel Consumption City (L/100km)', 'Fuel Consumption Hwy (L/100km)', 'Fuel Consumption Comb (L/100km)', 'Fuel Consumption Comb (mpg)'] + list(ohe.categories_[0])
X_train = X_train_all[input_variable]
X_test = X_test_all[input_variable]
    
linearModel = lm.LinearRegression()
linearModel.fit(X_train, y_train)

y_test_p = linearModel.predict(X_test)

plt.scatter(y_test, y_test_p, s = 15)
plt.title('Odnos izmedju stvarnih vrijednosti izlazne velicine i procjene dobivene modelom')
plt.xlabel('stvarne vrijednosti izlazne velicine')
plt.ylabel('procjena izlazne velicine')
plt.show()
print('Dijagram rasprsenja stvarnih vrijednosti izlazne velicine i procjene dobivene modelom u oba zadatka je linearan.')

ME = max_error(y_test, y_test_p)
print('Maksimalna pogreska u procjeni emisije C02 plinova u g/km je:',ME)
print('Radi se o vozilu:', X_test_all[abs(y_test-y_test_p) == ME]['Model'].iloc[0])
