import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)



plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap = matplotlib.colors.ListedColormap('red') , label = 'Train data')
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, marker='x', cmap = matplotlib.colors.ListedColormap('blue'), label = 'Test data')
plt.legend()
plt.show()

LogRegression_model = LogisticRegression()
LogRegression_model.fit(X_train, y_train)



colors = ['green', 'black']
# granica odluke u ravnini x1−x2 definirana je kao krivulja: a+bx1+cx2 = 0
# Izvlacenje parametara modela
theta0 = LogRegression_model.intercept_[0]
theta1, theta2 = LogRegression_model.coef_[0]
# Konstruiranje granice odluke
x_min = np.min(X_train[:, 1]) - 1
x_max = np.max(X_train[:, 1]) + 1
x1_values = np.linspace(x_min, x_max, 100)
x2_values = -theta0/theta2 -(theta1*x1_values)/theta2
#np.linspace generira niz od 100 tocaka koje su ravnomjerno rasporedjene unutar raspona minimalne i maksimalne vrijednosti prve znacajke u skupu
#za treniranje. Ovaj niz tocaka koristi se kao vrijednosti za varijablu x1 u daljnjem izracunu granice odluke 
# Prikazivanje granice odluke na scatter dijagramu
plt.plot(x1_values, x2_values)
plt.fill_between(x1_values, x2_values, x_min, color='red', alpha=0.2) 
plt.fill_between(x1_values, x2_values, x_max, color='green', alpha=0.2) 
# Prikazivanje podataka za treniranje na scatter dijagramu
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=matplotlib.colors.ListedColormap(colors), edgecolor="white")
plt.title('Granica odluke na podacima za treniranje')
plt.xlim(x_min, x_max)
plt.ylim(x_min, x_max)
plt.show()



# predikcija na skupu podataka za testiranje
y_test_p = LogRegression_model.predict(X_test)
# matrica zabune
cm = confusion_matrix(y_test, y_test_p)
print("\nd)")
print("Matrica zabune:\n" ,cm)
disp = ConfusionMatrixDisplay(confusion_matrix(y_test , y_test_p))
disp.plot()
plt.show()
print("Tocnost:" ,accuracy_score(y_test, y_test_p))
print("Preciznost:" ,precision_score(y_test, y_test_p))
print("Odziv:" ,recall_score(y_test, y_test_p))
# report
print("Izvjestaj\n",classification_report(y_test, y_test_p))



colors = ['green', 'black']
plt.scatter(X_test[:,0], X_test[:,1], c=y_test==y_test_p, cmap = matplotlib.colors.ListedColormap(colors))
cbar = plt.colorbar(ticks = [0,1])
cbar.set_ticklabels(['Tocno', 'Netocno'])
plt.title('Dobro i pogresno klasificirani testni primjeri')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
