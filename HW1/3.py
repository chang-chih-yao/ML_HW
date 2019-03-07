import numpy as np
import math
from matplotlib import pyplot as plt
sum = 0.0
prior_1 = [1/11, 1/11, 1/11, 1/11, 1/11, 1/11, 1/11, 1/11, 1/11, 1/11, 1/11]
prior_2 = [0.01, 0.01, 0.05, 0.08, 0.15, 0.4, 0.15, 0.08, 0.05, 0.01, 0.01]
prior_3 = [1/11, 1/11, 1/11, 1/11, 1/11, 1/11, 1/11, 1/11, 1/11, 1/11, 1/11]
combin = [1, 10, 45, 120, 210, 252, 210, 120, 45, 10, 1]
n = np.arange(0, 1.1, 0.1)

likelihood_arr = np.array([])
posterior_arr = np.array([])

for i in np.arange(0, 11):
    sum += (math.pow(i/10, 2) * math.pow((10-i)/10, 8) * prior_1[i])
#print("p(x):", sum)

#print("Likelihood:")
for i in np.arange(0, 11):
    temp = math.pow(i/10, 2) * math.pow((10-i)/10, 8) * combin[2]
    likelihood_arr = np.append(likelihood_arr, temp)
    #print(temp)

#print("posterior:")
for i in np.arange(0, 11):
    temp = math.pow(i/10, 2) * math.pow((10-i)/10, 8) * prior_1[i] / sum
    posterior_arr = np.append(posterior_arr, temp)
    #print(temp)



plt.figure(0)
plt.title("(a)")
plt.axis([-0.1,1.1,0,1])
plt.bar(n, prior_1, label='prior', width = 0.03, lw=1)
plt.legend(loc="upper left")
plt.show()

plt.figure(1)
plt.title("(a)")
plt.axis([-0.1,1.1,0,1])
plt.bar(n, posterior_arr, label='posterior', width = 0.03, lw=1)
plt.legend(loc="upper left")
plt.show()


sum = 0.0
posterior_arr2 = np.array([])
for i in np.arange(0, 11):
    sum += (math.pow(i/10, 2) * math.pow((10-i)/10, 8) * prior_2[i])
#print("p(x):", sum)


#print("posterior:")
for i in np.arange(0, 11):
    temp = math.pow(i/10, 2) * math.pow((10-i)/10, 8) * prior_2[i] / sum
    posterior_arr2 = np.append(posterior_arr2, temp)
    #print(temp)


plt.figure(4)
plt.title("(b)")
plt.axis([-0.1,1.1,0,1])
plt.bar(n, prior_2, label='prior', width = 0.03, lw=1)
plt.legend(loc="upper left")
plt.show()

plt.figure(2)
plt.title("(b)")
plt.axis([-0.1, 1.1, 0, 1])
plt.bar(n, posterior_arr2, label='posterior', width = 0.03, lw=1)
plt.legend(loc="upper left")
plt.show()


plt.figure(3)
plt.title("likelihood")
plt.axis([-0.1, 1.1, 0, 1])
plt.bar(n, likelihood_arr, label='likelihood', width = 0.03, lw=1)
plt.legend(loc="upper left")
plt.show()




entropy_arr = np.array([])
for iteration in range(50):

    cou = 0
    A = np.random.choice([0, 1], 10, p=[0.3, 0.7])
    #print("Distribution:", A)
    for i in range(10):
        if A[i] == 1:
            cou += 1


    sum = 0.0
    posterior_arr3 = np.array([])
    for i in np.arange(0, 11):
        sum += (math.pow(i/10, cou) * math.pow((10-i)/10, 10-cou) * prior_3[i])
    #print("p(x):", sum)


    for i in np.arange(0, 11):
        temp = math.pow(i/10, cou) * math.pow((10-i)/10, 10-cou) * prior_3[i] / sum
        posterior_arr3 = np.append(posterior_arr3, temp)
        #print(temp)
    prior_3 = posterior_arr3

    entropy = 0
    for i in range(11):
        entropy += posterior_arr3[i] * math.log(posterior_arr3[i] + 1e-9, 2)
    entropy *= -1
    entropy_arr = np.append(entropy_arr, entropy)

    if iteration%10==0:
        plt.figure(5)
        plt.title("(2)")
        plt.axis([0, 1, 0, 1])
        plt.bar(n, posterior_arr3, label='posterior', width = 0.03, lw=1)
        plt.legend(loc="upper left")
        plt.show()

n = np.arange(0, 50, 1)

plt.figure(6)
plt.title("Bonus")
#plt.axis([0, 1, 0, 1])
plt.plot(n, entropy_arr, label='entropy')
plt.legend(loc="upper left")
plt.show()
