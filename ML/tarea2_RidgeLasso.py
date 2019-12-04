# Mariana Hernandez 150845
# Tarea 2
# alias python='python3'
# pip3 install numpy

import numpy as np
import pandas as pd
import ssl
from matplotlib import pyplot as plt

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context


def ridge(Xp,X,y,L):
	# vector beta que minimiza los residuos
	beta = np.dot(np.linalg.inv(np.dot(X.transpose(),X) + L*np.identity(X.shape[1])),np.dot(X.transpose(),y))
	yhat = np.dot(Xp,beta)
	return yhat

# entrega los datos de los casos con cancer
def datos(modo = 'entrena'):
	url ='https://web.stanford.edu/~hastie/ElemStatLearn/datasets/prostate.data'
	df = pd.read_csv(url, sep='\t')
	tag ='T'if modo =='entrena'else'F'
	df = df[df['train']==tag].drop(columns=['Unnamed: 0','train'])
	X_cols = [c for c in df.columns if not c is'lpsa']
	return df[X_cols], df['lpsa']

def soft(a, delta):
	return np.sign(a)*max(abs(a)-delta,0)

def lasso(Xp,X,y,L):
	# iniciar beta estimador de Ridge
	beta = np.dot(np.linalg.inv(np.dot(X.transpose(),X) + L*np.identity(X.shape[1])),np.dot(X.transpose(),y))
	a = np.zeros(X.shape[1])
	c = np.zeros(X.shape[1])
	nuevaBeta = np.zeros(X.shape[1])
	epsilon = 1
	while epsilon > 10**(-6):
		for j in xrange(0,X.shape[1]):
			sum1 = 0
			sum2 = 0
			for i in xrange(0,X.shape[0]):
				sum1 = sum1 + X[i][j]**2
				sum2 = sum2 + X[i][j]*(y[i]-np.dot(beta.transpose(),X[i,:])+beta[j]*X[i][j])
			a[j] = 2*sum1
			c[j] = 2*sum2
			nuevaBeta[j] = soft(c[j]/a[j], L/a[j])
		epsilon = np.linalg.norm(nuevaBeta - beta)
		print('epsilon: %10f'%(epsilon))
		beta = nuevaBeta
	yhat = np.dot(Xp,beta)
	return yhat

##### validacion #####

datos(modo='entrena')
X,y = datos(modo='entrena')
X_numpy = X.to_numpy()
y_numpy = y.to_numpy()
N = X_numpy.shape[0]
L = 1

# pocos datos, utilizo validacion cruzada
partes = 9 # el residuo es la 10ma parte

# caso con numero de partes del mismo tamanio
sizeParte = int(N/partes)

if (N%partes == 0):
	i = 0
	while (i < N):
		Xp = X_numpy[i:i+sizeParte,:]
		X = np.concatenate((X_numpy[0:i,:],X_numpy[i+sizeParte:N,:]),axis=0) 
		yp = y_numpy[i:i+sizeParte]
		y = np.concatenate((y_numpy[0:i],y_numpy[i+sizeParte:N]),axis=0) 
		
		# # Ridge
		# resRidge1 = ridge(Xp,X,y,L)
		# print(resRidge1)

		# # Lasso
		# resLasso1 = lasso(Xp,X,y,L)
		# print(resLasso1)

		i = i+sizeParte

# caso donde no es division exacta de partes
if (N%partes != 0):
	k = N%partes
	i = 0
	while (i < N-k):

		Xp = X_numpy[i:i+sizeParte,:]
		X = np.concatenate((X_numpy[0:i,:], X_numpy[i+sizeParte:N,:]),axis=0)
		yp = y_numpy[i:i+sizeParte]
		y = np.concatenate((y_numpy[0:i],y_numpy[i+sizeParte:N]),axis=0)
		print("=====") 

		# # Ridge
		# resRidge1 = ridge(Xp,X,y,L)
		# print(resRidge1)

		# Lasso
		resLasso1 = lasso(Xp,X,y,L)
		print(resLasso1)
		print("=====")
		i = i+sizeParte

	while (i < N):

		Xp = X_numpy[i:N,:]
		X = X_numpy[0:i,:]
		yp = y_numpy[i:N]
		y = y_numpy[0:i]
		print("*****") 

		# Ridge
		# resRidge1 = ridge(Xp,X,y,L)
		# print(resRidge1)

		# Lasso
		resLasso1 = lasso(Xp,X,y,L)
		print(resLasso1)
		print("*****")
		i = i+k



