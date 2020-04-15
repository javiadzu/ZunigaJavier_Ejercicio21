import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import numpy as np
from sklearn import  metrics
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Cargamos los datos, son los mismos utilizados en el ejercicio 10
numeros = skdata.load_digits()
target = numeros['target']
imagenes = numeros['images']
n_imagenes = len(target)

# Aplanamos las imágenes 
data = imagenes.reshape((n_imagenes, -1)) # para volver a tener los datos como imagen basta hacer data.reshape((n_imagenes, 8, 8))


# Vamos a hacer un split training test
scaler = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.5)



x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#Utilizamos la regresión logísticas
clf = LogisticRegression(
    C=50., penalty='l1', solver='saga', tol=0.1
)
clf.fit(x_train, y_train)

#Predecimos los números necesitados
y_pred=clf.predict(x_test)

#Los coeficientes ya los halla  el programa 
coef = clf.coef_.copy()

#Rearmamos la figura donde encontramos los betas para cada uno de los dígitos
#Son 64 betas (uno por pixel) por 10 dígitos. Y lo graficamos.
plt.figure(figsize=(10, 5))
scale = np.abs(coef).max()
for i in range(10):
    l1_plot = plt.subplot(2, 5, i + 1)
    l1_plot.imshow(coef[i].reshape(8, 8), interpolation='nearest',
                   vmin=-scale, vmax=scale)
    l1_plot.set_xticks(())
    l1_plot.set_yticks(())
    l1_plot.set_xlabel('Número %i' % i)
plt.suptitle('Betas para cada número identificado')
plt.savefig('coeficientes.png')


#Calculamos la matriz de confusión y la imprimimos
matrix_confu=confusion_matrix(y_test, y_pred )
print (matrix_confu)
#Graficamos la matriz de confusión y la guardamos
disp = metrics.plot_confusion_matrix(clf, x_test, y_test)
disp.figure_.suptitle("Matriz de confusión")
plt.savefig('confusion.png')