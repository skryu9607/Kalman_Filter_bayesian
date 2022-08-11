import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import csv

file_name = 'data/record1659591134.183895.csv'
f = open(file_name, 'r')

tt = []
true_angle = []
acc = []
gyro = []
acc_roll1 = []
acc_roll2 = []
acc_pitch1 = []
acc_pitch2 = []
rdr = csv.reader(f)
cnt = 0
for line in rdr:
    if cnt > 0:
        tt.append(float(line[0][7:]))
        true_angle.append([float(line[1][8:]),float(line[2]),float(line[3])])
        acc.append([float(line[5]),float(line[6]),float(line[7]),float(line[12]),float(line[13]),float(line[14])])
        gyro.append([float(line[8]),float(line[9]),float(line[10]),float(line[15]),float(line[16]),float(line[17][0:5])])
        acc_roll1.append(np.arctan(acc[cnt - 1][1] / acc[cnt - 1][2]))
        acc_roll2.append(np.arctan(acc[cnt - 1][4] / acc[cnt - 1][5]))
        acc_pitch1.append(np.arctan(acc[cnt - 1][0] /(acc[cnt - 1][1]**2+acc[cnt - 1][2]**2)))
        acc_pitch2.append(np.arctan(acc[cnt - 1][3] / (acc[cnt - 1][4] ** 2 + acc[cnt - 1][5] ** 2)))
    cnt += 1

dt = 0.01
A = [[1, dt,0,0], [0, 1,0,0],[0, 0,1,dt], [0, 0,0,1]]
Q = [[0.05, 0, 0, 0], [0, 0.001, 0, 0],[0, 0, 0.05, 0], [0, 0, 0, 0.001]]
H = [[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1],[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]]
R = np.eye(8)*0.1
x = [[0], [0], [0], [0]]
P = [[0.01, 0, 0, 0],[0, 0.01, 0, 0],[0, 0, 0.01, 0],[0, 0, 0, 0.01]]

x = np.asarray(x)
A = np.asarray(A)
P = np.asarray(P)
Q = np.asarray(Q)
H = np.asarray(H)

X_sav = []
dt = tt[0]
for i in range(len(tt)):
    if i > 0:
        dt = tt[i] - tt[i-1]
    A = [[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]]
    A = np.asarray(A)

    x_ = np.matmul(A, x)
    P_ = np.matmul(np.matmul(A, P), np.transpose(A)) + Q
    K = np.matmul(np.matmul(P_,np.transpose(H)),LA.inv(np.matmul(np.matmul(H,P_),np.transpose(H))+R))
    meas = [acc_roll1[i],gyro[i][0],acc_pitch1[i],gyro[i][1], acc_roll2[i],gyro[i][3],acc_pitch2[i],gyro[i][4]]
    meas = np.asarray(meas)
    meas = np.transpose(meas)

    x = x_ + np.transpose( np.matmul(meas - np.transpose(np.matmul(H,x_)),np.transpose(K)) )
    P = np.matmul((np.eye(4) - np.matmul(K,H)), P_)
    X_sav.append([x[0][0], x[1][0], x[2][0], x[3][0]])

dt = 0.01
A = [[1, dt,0,0], [0, 1,0,0],[0, 0,1,dt], [0, 0,0,1]]
Q = [[0.05, 0, 0, 0], [0, 0.001, 0, 0],[0, 0, 0.05, 0], [0, 0, 0, 0.001]]
H = [[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]]
R = np.eye(4)*0.1
x = [[0], [0], [0], [0]]
P = [[0.01, 0, 0, 0],[0, 0.01, 0, 0],[0, 0, 0.01, 0],[0, 0, 0, 0.01]]

x = np.asarray(x)
A = np.asarray(A)
P = np.asarray(P)
Q = np.asarray(Q)
H = np.asarray(H)

X_sav_ = []
dt = tt[0]
for i in range(len(tt)):
    if i > 0:
        dt = tt[i] - tt[i-1]
    A = [[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]]
    A = np.asarray(A)

    x_ = np.matmul(A, x)
    P_ = np.matmul(np.matmul(A, P), np.transpose(A)) + Q
    K = np.matmul(np.matmul(P_,np.transpose(H)),LA.inv(np.matmul(np.matmul(H,P_),np.transpose(H))+R))
    meas = [acc_roll1[i],gyro[i][0],acc_pitch1[i],gyro[i][1]]
    meas = np.asarray(meas)
    meas = np.transpose(meas)

    x = x_ + np.transpose( np.matmul(meas - np.transpose(np.matmul(H,x_)),np.transpose(K)) )
    P = np.matmul((np.eye(4) - np.matmul(K,H)), P_)
    X_sav_.append([x[0][0], x[1][0], x[2][0], x[3][0]])

plt.subplot(2,2,1)
plt.plot(tt,np.transpose(true_angle)[0])
plt.plot(tt,np.transpose(X_sav)[0]*180/np.pi)
plt.plot(tt,np.transpose(X_sav_)[0]*180/np.pi)
plt.legend(['true','2mpu','1mpu'])
plt.ylabel('Angle(deg)')
plt.title('Rolling Angle')

plt.subplot(2,2,2)
plt.plot(tt,-np.transpose(true_angle)[1])
plt.plot(tt,np.transpose(X_sav)[2]*180/np.pi)
plt.plot(tt,np.transpose(X_sav_)[2]*180/np.pi)
plt.legend(['true','2mpu','1mpu'])
plt.ylabel('Angle(deg)')

plt.title('Pitching Angle')

plt.subplot(2,2,3)
plt.plot(tt,np.abs(np.transpose(X_sav)[0]*180/np.pi - np.transpose(true_angle)[0]))
plt.plot(tt,np.abs(np.transpose(X_sav_)[0]*180/np.pi - np.transpose(true_angle)[0]))
plt.legend(['2mpu','1mpu'])
plt.xlabel('Time(sec)')
plt.ylabel('Angle(deg)')

plt.subplot(2,2,4)
plt.plot(tt,np.abs(np.transpose(X_sav)[2]*180/np.pi + np.transpose(true_angle)[1]))
plt.plot(tt,np.abs(np.transpose(X_sav_)[2]*180/np.pi + np.transpose(true_angle)[1]))
plt.legend(['2mpu','1mpu'])
plt.ylabel('Angle(deg)')
plt.xlabel('Time(sec)')
# plt.plot(tt,np.transpose(gyro)[0])
# plt.plot(tt,np.transpose(gyro)[1])
# plt.plot(tt,np.transpose(gyro)[2])
plt.show()