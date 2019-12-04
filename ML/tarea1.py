# Mariana Hernandez 150845
# Tarea 1

import numpy as np
import statistics
import time

# funcion que encuentra los k vecinos mas cercanos
# ToDo : si ordeno X, puedo empezar a comparar por los puntos aledanios
# ToDo : sustituir for loops con while
def knn(x,X,y,k):
	x = x
	t = x.shape[0] # cant de puntos a comparar
	m = X.shape[0] # cant total de puntos
	vecindario = np.full((1,k),0)
	#vecindario = np.array([[0,0,0,0]]) # vecinos para todos los puntos de x
	yhat = np.full((1,k),0)

	# comparar la distancia contra todos los puntos
	j = 0
	while j < t:

		dummy = np.linalg.norm(x[j,:]-X[0,:])
		vecinos = np.array([dummy]) # vecinos para el punto ti
		etiquetas = np.array([])

		# encontrar los knn
		i = 1
		while i < m:
			
			dist = np.linalg.norm(x[j,:]-X[i,:])

			if vecinos.shape[0] < k:

				vecinos = np.append(vecinos, [dist])
				etiquetas = np.append(etiquetas,y[i])
				i = i + 1

			if vecinos.shape[0] == k and np.amax(vecinos) > dist:

				ind = np.where(vecinos == np.amax(vecinos))[0]
				vecinos = np.delete(vecinos, ind)
				etiquetas = np.delete(etiquetas, ind)
				vecinos = np.append(vecinos, [dist])
				etiquetas = np.append(etiquetas,y[i])
				i = i + 1

			else:
				i = i + 1

		vecindario = np.vstack((vecindario, vecinos)) 
		#yhat = np.vstack((yhat,etiquetas)) 
		yhat = np.append(yhat, [etiquetas], axis = 0)
		vecinos = np.array([])
		j = j+1

	# predecir etiqueta para cada punto en x
	np.asarray(yhat)
	prediccion = np.array([0])

	for x in xrange(1,t+1):
			
		try:
			prediccion = np.append(prediccion,[statistics.mode(yhat[x,:])])
		except statistics.StatisticsError:
			prediccion = np.append(prediccion,0) # Ambas etiquetas son posibles
	prediccion = np.delete(prediccion,0) # elimino cero del inicio

	return prediccion # yhat

# funcion que genera datos de entrada para knn
def datos(N, modo = 'entrena'):
	seed = 100 if modo == 'entrena' else 200
	np.random.seed(seed)
	n1 = int(N/2)
	n2 = N - n1
	cov = [[1,0],[0,1]]
	X1 = np.random.multivariate_normal([3,1],cov,n1)
	X2 = np.random.multivariate_normal([2,4],cov,n2)
	X = np.concatenate((X1,X2), axis = 0)
	y = np.concatenate((np.ones(n1),2*np.ones(n2)), axis = 0)
	Xy = np.concatenate((X,np.expand_dims(y, axis = 1)), axis = 1)
	np.random.shuffle(Xy)
	return Xy[:,[0,1]], Xy[:,2].squeeze()

# validacion
n = 300 
Conj_e = int(n*0.8)
Conj_p = int(n*0.2)
X, y = datos(n)

X_e = X[0:Conj_e,:]
y_e = y[0:Conj_e]
x_p = X[Conj_e:n,:]
y_p = y[Conj_e:n]
k = 7

start = time.time() # para obtener tiempo de ejecucion
res = knn(x_p, X_e, y_e, k) # comparar contra y_p 
end = time.time()
tiempo = end - start
print(tiempo)

# comprobacion del experimento
comprobacion = np.equal(res,y_p)
pAciertos = (float(np.sum(comprobacion))/float(res.shape[0]))*100
pError = ((float(res.shape[0])-float(np.sum(comprobacion)))/float(res.shape[0]))*100

