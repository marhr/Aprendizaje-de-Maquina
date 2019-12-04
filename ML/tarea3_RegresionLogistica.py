# Mariana Hernandez 150845
# Tarea 3

import numpy as np
import pandas as pd


# regresion logistica

def sigmoide(z):
	return 1/(1+np.exp(-z))

def funcion_perdida(mu, y):
	return (1/len(y))*((np.dot((-y).T,np.log(mu)))-(np.dot((1-y).T,np.log(1-mu))))
	#return (1/len(y))*(((-y).T @ np.log(mu))-((1-y).T @ np.log(1-mu)))

def grad_F(X, y, beta):
	return np.dot(X.T, (sigmoide(np.dot(X, beta)) - y) )

def hess_F(X, y, beta):
	S = np.dot(sigmoide(np.dot(X, beta)),(1-sigmoide(np.dot(X, beta))))
	return np.dot(np.dot(X.T, S),X)

# tao = 0.5
# gamma = 0.1
# epsilon = 10**(-8)
def maximo_descenso(X, y, beta, tao, gamma):
	paso = -grad_F(X, y, beta) # direccion:p
	alpha = 0.5 # longitud inicial
	iteraciones = 0
	condicion = True
	nuevaBeta = beta

	while (condicion):

		beta = nuevaBeta
		#print(beta)
		costo_viejo = funcion_perdida(sigmoide(np.dot(X, beta)), y)
		#print(costo_viejo)
		nuevaBeta = beta - alpha*paso
		#print(nuevaBeta)
		costo = funcion_perdida(sigmoide(np.dot(X, nuevaBeta)), y)
		#print(costo)
		iteraciones += 1
		alpha = alpha*tao

		condicion = (10**(-8) > np.linalg.norm(nuevaBeta - beta))#&(costo-costo_viejo >= -gamma*alpha*paso ) 
		

	return alpha, nuevaBeta, iteraciones

def newton(X, y, beta, tao, gamma):
	condicion = True

	while (condicion):
		
		condicion = (10**(-8) > np.linalg.norm(nuevaBeta - beta))
		
	pass

def pred(X, betahat):
	yhat = np.round(sigmoide(np.dot(X, betahat)))
	return yhat

def datos(modo='entrena'):
	gid = 'd932a3cf4d6bdeef36a7230fff959301'
	tail = '64b604aedff376b7757b533d1c93685ce19b2077/bcdata'
	url = 'https://gist.githubusercontent.com/rodrgo/%s/raw/%s' % (gid, tail)
	df = pd.read_csv(url, sep=',')
	df = df.drop(columns=['Unnamed: 32', 'id'])
	var = 'diagnosis'
	df.loc[df[var] == 'M', [var]] = 1
	df.loc[df[var] == 'B', [var]] = 0
	X_cols = [c for c in df.columns if not c is var]
	X, y = df[X_cols].to_numpy(), df[var].to_numpy()
	idx = np.random.permutation(X.shape[0])
	train_idx, test_idx = idx[:69], idx[69:]
	idx = train_idx if modo == 'entrena' else test_idx
	return X[idx,:], y[idx]


### V A L I D A C I O N ###

X_entrena, y_entrena = datos()
print(X_entrena[0])
print(y_entrena)
X_prueba, y_prueba = datos(modo='prueba')
# beta Ridge
#L = 1
betahat = np.dot(np.linalg.inv(np.dot(X_entrena.transpose(),X_entrena) + L*np.identity(X_entrena.shape[1])),np.dot(X_entrena.transpose(),y_entrena))
#maximo_descenso(X_entrena, y_entrena, betahat, 0.5, 0.1)




