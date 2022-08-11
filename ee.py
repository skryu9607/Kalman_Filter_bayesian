import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import csv

file_name = 'data/record1659591033.904333.csv'
f = open(file_name, 'r')

tt = []
true_angle = []
acc = []
gyro = []
acc_roll1 = []
acc_roll_d = []
d1 = []
d2 = []
rdr = csv.reader(f)
cnt = 0
for line in rdr:
    if cnt > 0:
        # print(line)
        # print("time", float(line[0][7:]))
        aa = line[1].split(" ")
        # print("true angle", float(aa[1][6:]), float(aa[2]), float(aa[3]))
        # print("acc", float(aa[7]), float(line[2]), float(line[3]))
        # print("gyro", float(line[4][9:]), float(line[5]), float(line[6]))
        # print("d1", int(line[8][5:]))
        # print("d2", int(line[10][5:-6]))

        tt.append(float(line[0][7:]))
        true_angle.append([float(aa[1][6:]), float(aa[2]), float(aa[3])])
        acc.append([float(aa[7])/9.81, float(line[2])/9.81, float(line[3])/9.81])
        d1.append(int(line[8][5:]))
        d2.append(int(line[10][5:-5]))
        gyro.append([float(line[4][9:])/180*np.pi, float(line[5])/180*np.pi, float(line[6])/180*np.pi])
        acc_roll1.append(np.arctan(acc[cnt - 1][1] / acc[cnt - 1][2]))
        acc_roll_d.append(np.arctan((d1[cnt-1]-d2[cnt-1])/52))
    cnt += 1
#
dt = 0.01
A = [[1, dt], [0, 1]]
Q = [[0.001, 0], [0, 0.001]]
R = [[0.1,0,0],[0,0.01,0],[0,0,0.1]]
H = [[1, 0],[0, 1],[1, 0]]
x = [[0], [0]]
P = [[0.1, 0],[0, 0.1]]

x = np.asarray(x)
A = np.asarray(A)
P = np.asarray(P)
Q = np.asarray(Q)
H = np.asarray(H)
#
X_sav = []
dt = tt[0]
for i in range(len(tt)):
    if i > 0:
        dt = tt[i] - tt[i-1]
    A = [[1, dt], [0, 1]]
    A = np.asarray(A)

    x_ = np.matmul(A, x)
    P_ = np.matmul(np.matmul(A, P), np.transpose(A)) + Q
    K = np.matmul(np.matmul(P_,np.transpose(H)),LA.inv(np.matmul(np.matmul(H,P_),np.transpose(H))+R))
    meas = [acc_roll1[i],gyro[i][0],acc_roll_d[i]]
    meas = np.asarray(meas)
    meas = np.transpose(meas)

    x = x_ + np.transpose( np.matmul(meas - np.transpose(np.matmul(H,x_)),np.transpose(K)) )
    P = np.matmul((np.eye(2) - np.matmul(K,H)), P_)
    X_sav.append([x[0][0], x[1][0]])
#
dt = 0.01
R = [[0.1,0],[0,0.01]]
H = [[1, 0],[0, 1]]

x = [[0], [0]]
P = [[0.1, 0],[0, 0.1]]

x = np.asarray(x)
A = np.asarray(A)
P = np.asarray(P)

H = np.asarray(H)
#
X_sav_ = []
dt = tt[0]
for i in range(len(tt)):
    if i > 0:
        dt = tt[i] - tt[i-1]
    A = [[1, dt], [0, 1]]
    A = np.asarray(A)

    x_ = np.matmul(A, x)
    P_ = np.matmul(np.matmul(A, P), np.transpose(A)) + Q
    K = np.matmul(np.matmul(P_,np.transpose(H)),LA.inv(np.matmul(np.matmul(H,P_),np.transpose(H))+R))
    meas = [acc_roll1[i],gyro[i][0]]
    meas = np.asarray(meas)
    meas = np.transpose(meas)

    x = x_ + np.transpose( np.matmul(meas - np.transpose(np.matmul(H,x_)),np.transpose(K)) )
    P = np.matmul((np.eye(2) - np.matmul(K,H)), P_)
    X_sav_.append([x[0][0], x[1][0]])
#
plt.subplot(2,1,1)
plt.plot(tt,np.transpose(true_angle)[0])
plt.plot(tt,np.transpose(X_sav)[0]*180/np.pi)
plt.plot(tt,np.transpose(X_sav_)[0]*180/np.pi)
plt.legend(['true','+dist_sensor','1mpu'])
plt.ylabel('Angel(deg)')

plt.subplot(2,1,2)
plt.plot(tt,np.abs(np.transpose(X_sav)[0]*180/np.pi - np.transpose(true_angle)[0]))
plt.plot(tt,np.abs(np.transpose(X_sav_)[0]*180/np.pi - np.transpose(true_angle)[0]))
plt.legend(['+dist_sensor','1mpu'])
plt.xlabel('time(sec)')
plt.ylabel('Angel(deg)')
# # plt.plot(tt,np.transpose(gyro)[0])
# # plt.plot(tt,np.transpose(gyro)[1])
# # plt.plot(tt,np.transpose(gyro)[2])
# plt.plot(tt,np.transpose(acc_roll1))
# plt.plot(tt,np.transpose(acc_roll_d))

plt.show()