# Mariana Hernandez 150845
# Tarea 4

import numpy as np
import math
from scipy import special
import matplotlib.pyplot as plt

def cholesky(A):
	# R = A
	# Algoritmo de Rodrigo
	# for k in range(n):
	# 	for j in range(k + 1):
	# 		R[j,j:n] = R[j,j:n] - ((R[k,j]/R[k,k])*R[k,j:n])
	# 		#((R[k,j]/R[k,k])*(R[j,j:n] - R[k,j:n]))
	# 	R[k,k:n] = 1/(R[k,k]**(1/2))*R[k,k:n]
	# return R

	# Mi algoritmo
	n = A.shape[0]
	L = np.zeros_like(A)
	for row in range(n):
		for col in range(row+1):
			suma = np.dot(L[row,:col],L[col,:col].T)
			if row == col:
				L[row,col]=math.sqrt(max(A[row,row]-suma,0))
			else:
				L[row,col]=(1.0/L[col,col])*(A[row,col]-suma)
	return L

def generadorVA(media=0, var = 1, n=100):
	x =(media+2**(1/2))*var*special.erfinv(np.random.uniform(size=n)*2-1)
	return x

def kernel(x1, x2, alpha=1, beta=0.1):
	return alpha*math.exp(-(np.linalg.norm(x1-x2,'fro'))/(2*(beta)))

### V A L I D A C I O N ###

# A = np.matrix([[25, 15, -5, -10],[15, 10, 1, -7],[-5, 1, 21, 4],[-10, -7, 4, 18]])
# cholesky(A)
# va = generadorVA()

# Funcion a aproximar
f = lambda x: np.sin(0.9*x).flatten()
# Datos de entrenamiento:
N = 10
sigma = 0.00005
X = np.random.uniform(-5, 5, size=(N,1))
y = f(X) + sigma*np.random.randn(N)
# Puntos a predecir
m = 50
Xprueba = np.linspace(-5, 5, m).reshape(-1,1)

# obteniendo mu, s2
k = kernel(X,X) * np.ones((Xprueba.shape[0],Xprueba.shape[0]))
k_e = kernel(X,Xprueba.T) * np.ones((Xprueba.shape[0],X.shape[0]))
k_ee = kernel(Xprueba,Xprueba) * np.ones((X.shape[0],X.shape[0]))
print(k_ee.shape)
mu = np.dot(np.dot(k_e,np.linalg.inv(k)),y)
s = -np.dot(np.dot(k_e.T,np.linalg.inv(k)),k_e)+k_ee

# graficas intervalos de confianza
plt.figure(1)
plt.clf()
plt.plot(X, y, 'r+', ms=20)
plt.plot(Xprueba, f(Xprueba), 'b-')
plt.gca().fill_between(Xprueba.flat, mu-3*s, mu+3*s, color="#dddddd")
plt.plot(Xprueba, mu, 'r--', lw=2)
plt.title('Promedio de las predicciones mas/menos 3 deviaciones estandar')
plt.axis([-5, 5, -3, 3])
plt.savefig('predictiva.png', bbox_inches='tight')

# distribucion a priori
plt.figure(2)
plt.clf()
plt.plot(Xprueba, f_prior)
plt.title('Diez muestras de la priori de Procesos Gaussianos')
plt.axis([-5, 5, -3, 3])
plt.savefig('prior.png', bbox_inches='tight')

# distribucion posterior
plt.figure(3)
plt.clf()
plt.plot(Xprueba, f_posterior)
plt.title('Diez muestras de la posterior de Procesos Gaussianos')
plt.axis([-5, 5, -3, 3])
plt.savefig('posterior.png', bbox_inches='tight')
plt.show()













