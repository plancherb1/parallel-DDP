from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

f = open('t.txt','r')
data = f.readlines()
x_1 = np.array([])
u_1 = np.array([])
for row in data:
    stamp, rest = row.split(']')
    q,qd,u = rest.split('|')
    qs = q.split(' ')[1:-1]
    qds = qd.split(' ')[1:-1]
    us = u.split(' ')[1:]
    us[-1] = us[-1][:-1]
    try:
    	x_1 = np.vstack((x_1, np.array(qs + qds)))
    	u_1 = np.vstack((u_1, np.array(us)))
    except:
    	x_1 = np.array(qs + qds)
    	u_1 = np.array(us)
x_1 = x_1.astype(float)
u_1 = u_1.astype(float)
t_1 = range(x_1.shape[0])

f2 = open('t2.txt','r')
data2 = f2.readlines()
x_2 = []
u_2 = []
u_2m = []
for row in data2:
    stamp,q,qd,u,um = row.split('|')
    qs = q.split(' ')[1:-1]
    qds = qd.split(' ')[1:-1]
    us = u.split(' ')[1:-1]
    ums = um.split(' ')[1:]
    ums[-1] = ums[-1][:-1]
    try:
    	x_2 = np.vstack((x_2, np.array(qs + qds)))
    	u_2 = np.vstack((u_2, np.array(us)))
    	u_2m = np.vstack((u_2m, np.array(ums)))
    except:
    	x_2 = np.array(qs + qds)
    	u_2 = np.array(us)
    	u_2m = np.array(ums)
x_2 = x_2.astype(float)
u_2 = u_2.astype(float)
u_2m = u_2m.astype(float)
t_2 = range(x_2.shape[0])

index = 4
index2 = 5
plt.rc_context({'axes.edgecolor':'#222222'})
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111)
ax.scatter(t_1[::100],u_1[::100,index], c ='r', alpha = 0.8)
ax.scatter(t_2[:-3500:100],-u_2[3500::100,index], c ='b', alpha = 0.8)
ax.scatter(t_1[::100],u_1[::100,index2], c ='g', alpha = 0.8)
ax.scatter(t_2[:-3500:100],-u_2[3500::100,index2], c ='y', alpha = 0.8)
plt.show()