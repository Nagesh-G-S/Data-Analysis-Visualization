import numpy as np
import csv
import scipy.stats as sst
import random
import math
import matplotlib.pyplot as plt






def sparse_sovler(A, y, gamma):
	N = len(y)
	D = len(A[0])
	x = np.zeros((D, 1))
	theta = np.zeros((D, 1))
	active_set = np.empty((0), dtype = int)

	count_2 = 0
	while(1):
		#step2
		d = -2 * np.matmul(A.T, y) + 2 * reduce(np.matmul, [A.T, A, x])
		max_idx = np.argmax(np.absolute(d));
		active_set = np.append(active_set, [max_idx])
		active_set = np.unique(active_set)
		if d[max_idx] > gamma:
			theta[max_idx] = -1.0
		elif d[max_idx] < -gamma:
			theta[max_idx] = 1.0;

		
		count_3 = 0

		
		while(1):
			#step3
			A_hat = A[:, active_set]
			x_hat = x[active_set]
			theta_hat = theta[active_set]
			x_hat_new = np.matmul(np.linalg.inv(np.matmul(A_hat.T, A_hat)), np.matmul(A_hat.T, y) - (gamma / 2.0) * theta_hat)
			sign_change_indx = np.argwhere((np.sign(x_hat) * np.sign(x_hat_new)).reshape(-1) == -1)
			x_all_new = np.ones(len(sign_change_indx) + 1) * x_hat_new
			x_all = np.ones(len(sign_change_indx) + 1) * x_hat	
			alpha = - x_hat_new[sign_change_indx] / (x_hat[sign_change_indx]- x_hat_new[sign_change_indx])
			alpha = np.append(0, alpha.T)
			x_tocheck = x_all * alpha + x_all_new * (1 - alpha)
			obj_fun = np.power(np.linalg.norm(y - np.matmul(A_hat, x_tocheck), axis = 0),2) + gamma * np.matmul(theta_hat.T, x_tocheck)
			min_idx = np.argmin(obj_fun)
			x_new = x_tocheck[:,min_idx]
			active_set = active_set[np.absolute(x_new)> 1e-6]
			x_new = x_new[np.absolute(x_new) > 1e-6]
			x[active_set] = x_new.reshape((-1,1))
			step3_flag =  -2 * np.matmul(A.T, y) + 2 * reduce(np.matmul, [A.T, A, x]) + gamma * np.sign(x) 
			if (np.absolute(step3_flag[np.absolute(x) > 1e-9]) < 1e-9).all():
				break
			count_3 = count_3 + 1

		step2_flag = -2 * np.matmul(A.T, y) + 2 * reduce(np.matmul, [A.T, A, x])
		if (np.absolute(step2_flag[np.absolute(x) < 1e-9]) <= gamma).all():
			print("i am breaking out")
			break
		count_2 = count_2 + 1
	return (x)



#sizes
N = 4
D = N

#vectors and matrices
y = np.random.rand(N, 1)
A = np.random.rand(N, D)
gamma = 1e-8;
#y = np.array([[-1],[2],[-3],[4]])
#A = np.identity(4)
#y = np.array([[1],[2],[3],[4]])
#A = np.array([[2.0,3.0,4.0,8.0], [1.0,5.0,9.0,2.0], [4.0,7.0,2.0,9.0], [1.0,9.0,3.0,7.0]])

x = sparse_sovler(A, y, gamma)




y_hat = np.matmul(A,x)
print "x : ", x
print "A : ", A
print "y (original) : ", y
print"y_hat = A*x : ", y_hat