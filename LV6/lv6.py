import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV


from sklearn.model_selection import cross_val_score

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
                    label=cl)


# ucitaj podatke
data = pd.read_csv("Social_Network_Ads.csv")
print(data.info())

data.hist()
plt.show()

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

# Model logisticke regresije
LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_train_n, y_train)

# Evaluacija modela logisticke regresije
y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)

print("Logisticka regresija: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

# granica odluke pomocu logisticke regresije
plot_decision_regions(X_train_n, y_train, classifier=LogReg_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()


#1
# inicijalizacija i ucenje KNN modela
KNN_model = KNeighborsClassifier(n_neighbors = 5)#K=5, 100, 1
KNN_model.fit(X_train_n, y_train)

# Evaluacija KNN modela 
y_train_p_KNN = KNN_model.predict(X_train_n)
y_test_p_KNN = KNN_model.predict(X_test_n)

# tocnost klasifikacije na skupu podataka za ucenje i skupu podataka za testiranje
print("\nAlgoritam KNN: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p_KNN))))

#zakljucak
#print("Vezano uz dobivenu granicu odluke KNN modela, zakljucujem da je veca tocnost KNN modela od logisticke regresije.")

#2
# granica odluke pomocu KNN modela
plot_decision_regions(X_train_n, y_train, classifier=KNN_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost(KNN model): " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN))))
plt.tight_layout()
plt.show()

#zakljucak-granica odluke
print("Granica odluke kada je K=1 je overfitting, K=100 underfitting, a za K=5 je optimalna.")

#Zadatak 6.5.2
scores = cross_val_score(KNN_model, X_train_n, y_train, cv =5)
print(scores)

#Zadatak 6.5.3
# inicijalizacija i ucenje SVM modela
SVM_model = svm.SVC(kernel ='rbf', gamma = 1, C=0.1)#mijenjati kernel, C i gamma
SVM_model.fit(X_train_n, y_train)
# Evaluacija SVM modela 
y_train_p_SVM = SVM_model.predict(X_train_n)
y_test_p_SVM = SVM_model.predict(X_test_n)
# tocnost klasifikacije na skupu podataka za ucenje i skupu podataka za testiranje
print("\nSVM: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_SVM))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p_SVM))))
# granica odluke pomocu SVM modela
plot_decision_regions(X_train_n, y_train, classifier=SVM_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost(SVM model): " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_SVM))))
plt.tight_layout()
plt.show()

#Kako promjena ovih hiperparametara utjece na granicu odluke te pogresku na skupu podataka za testiranje?
#Mijenjajte tip kernela koji se koristi. Sto primjecujete?
# print("Smanjivanjem C i gamme, dogadja se underfitting, povecanjem overfitting.")
# print("Promjenom kernela za iste parametre granica odluke se mijenja.")

#Zadatak 6.5.4
param_grid = {'C': [1, 10, 100], 'gamma': [10, 1, 0.1, 0.01]}
svm_gscv = GridSearchCV (SVM_model, param_grid, cv =5, scoring ='accuracy', n_jobs =-1)
svm_gscv.fit(X_train_n, y_train)

print("\nNajbolji hiperparametri")
print(svm_gscv.best_params_)
print("\nNajbolja prosjecna vrijednost")
print(svm_gscv.best_score_)
print("\nRezultati pretrage po resetki")
print(svm_gscv.cv_results_)


