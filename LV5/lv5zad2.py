import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


labels= {0:'Adelie', 1:'Chinstrap', 2:'Gentoo'}

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    edgecolor = 'w',
                    label=labels[cl])

# ucitaj podatke
df = pd.read_csv("penguins.csv")

# izostale vrijednosti po stupcima
print(df.isnull().sum())

# spol ima 11 izostalih vrijednosti; izbacit cemo ovaj stupac
df = df.drop(columns=['sex'])

# obrisi redove s izostalim vrijednostima
df.dropna(axis=0, inplace=True)

# kategoricka varijabla vrsta - kodiranje
df['species'].replace({'Adelie' : 0,
                        'Chinstrap' : 1,
                        'Gentoo': 2}, inplace = True)

print(df.info())

# izlazna velicina: species
output_variable = ['species']

# ulazne velicine: bill length, flipper_length
input_variables = ['bill_length_mm',
                    'flipper_length_mm']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()

# podjela train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)



#a)
classes, count_train = np.unique(y_train, return_counts=True)
classes, count_test = np.unique(y_test, return_counts=True)

plt.bar(classes-0.2, count_train, color=['blue'], width = 0.4, label='Train')
plt.bar(classes+0.2, count_test, color=['yellow'], width = 0.4, label='Test')
plt.title('Number of examples for each class(train and test data)')
plt.xlabel('Classes-Penguins')
plt.ylabel('Counts')
plt.legend()
plt.xticks((np.arange(len(classes))), ['Adelie(0)','Chinstrap(1)','Gentoo(2)'])
plt.show()

#b)
# inicijalizacija i ucenje modela logisticke regresije
LogRegression_model = LogisticRegression(max_iter=120)
LogRegression_model.fit(X_train , y_train)

#c)
theta0 = LogRegression_model.intercept_
theta12 = LogRegression_model.coef_
print("\nc)")
print(theta0)
print(theta12)
# Ima 3 retka i 2 stupca zbog broja klasa (3), svaki redak za jednu klasu. Kod binarne klasifikacije je bio 1 red s 2 stupca.")
# Svaki stupac u paru s jednom ulaznom velicinom")
# To je zato sto logisticka regresija prilagodjava viseklasni problem tako da stvara binarne klasifikatore za svaku klasu u odnosu na ostale 
# klase. Dakle, dobivamo po jedan set koeficijenata (parametara) za svaku klasu.
# U ovom slucaju, izlaz su tri klase (jer postoji tri vrijednosti u theta0).

#d)
# y_train_tsposed = np.transpose(y_train)[0]
# plot_decision_regions(X_train, y_train_tsposed, LogRegression_model)

# plot_decision_regions(X_train, y_train, LogRegression_model)
# print("\nd)")
# print("Vizuelno pregledom grafa rezultata multinomijalne logisticke regresije vidimo da je model dobro naucen.")
# print("To zakljucujemo jer su klase dobro razdvojene na grafu i tocke iste klase grupirane zajedno.")
# print("Nema puno preklapanja.")
     
#e)
# predikcija na skupu podataka za testiranje ili provodjenje klasifikacije skupa podataka za testiranje pomocu izgradjenog modela logisticke
#regresije.
y_test_p = LogRegression_model.predict(X_test)

# matrica zabune
cm = confusion_matrix(y_test, y_test_p)
print("\ne)")
print("Matrica zabune:\n" ,cm)
disp = ConfusionMatrixDisplay(confusion_matrix(y_test , y_test_p))
disp.plot()
plt.show()

print("Tocnost:" ,accuracy_score(y_test, y_test_p))

# report
print("Izvjestaj\n",classification_report(y_test , y_test_p))

#f)
input_variables2 = ['bill_length_mm',
                    'bill_depth_mm']

X_new = df[input_variables2].to_numpy()
y_new = df[output_variable].to_numpy()
y_new = y_new.ravel()#dodamo jer je javljao gresku da je y bio 2d

X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size = 0.2, random_state = 123)

LogRegression_model2 = LogisticRegression(max_iter=120)
LogRegression_model2.fit(X_train_new , y_train_new)

plot_decision_regions(X_train_new, y_train_new, LogRegression_model2)
y_test_p_new = LogRegression_model.predict(X_test_new)

# Dodavanje dodatnih ulaznih varijabli moze rezultirati boljim performansama modela jer model ima vise informacija na raspolaganju za donosenje 
# odluka. U nekim slucajevima dodavanje dodatnih ulaznih varijabli moze imati malen ili nikakav utjecaj na performanse modela, posebno ako su te 
# varijable vec vrlo slicne. 
