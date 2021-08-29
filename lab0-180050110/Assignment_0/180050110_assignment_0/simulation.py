import random
import matplotlib.pyplot as plt
from math import sqrt
p = 0.75
num = 2

def simulate_event():
	state = 0
	count = 0
	while state != num:
		count+=1
		v = random.random()
		if v <= p:
			state += 1
		else:
			state = 0
	return count 

def simulate_series(n):
	total = 0
	for i in range(n):	
		total += simulate_event()
	return total/n

def simulate_round(n):
	m = 10**n 
	y = []
	x = []
	for i in range(10):
		y.append(simulate_series(m))
		x.append(m)
	return x,y

if __name__ == "__main__":

	for i in range(1,5):
		x,y = simulate_round(i)
		# plt.plot(x, y, 'rx')
		mean = sum(y)/len(y)
		sigma = sqrt(sum([(i-mean)**2 for i in y]))/len(x)
		plt.plot(x[0:1],mean,'x')
		plt.errorbar(x[0:1],mean,yerr = sigma)
		print(sigma)
	plt.xscale('log')
	plt.xlabel('n')
	plt.ylabel('expected value')
	plt.title('q3')
	plt.show()

