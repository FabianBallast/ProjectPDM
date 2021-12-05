import matplotlib.pyplot as plt
import numpy as np  
import mpl_toolkits.mplot3d.axes3d as p3

# Select one
# from util.hover import hover as traj
# from util.circle import circle as traj
# from util.diamond import diamond as traj
from util.tud import tud as traj

T = 20
dt = 0.1
t = np.linspace(0, T, int(T/dt)+1)

pos = np.zeros((len(t), 3))
vel = np.zeros_like(pos)
acc = np.zeros_like(pos)

for idx, time in enumerate(t):
    p, v, a, y, yd = traj(time)
    pos[idx, :] = p
    vel[idx, :] = v
    acc[idx, :] = a

plt.figure(1)
plt.subplot(3,1,1)
plt.plot(t, pos)  

plt.subplot(3,1,2)
plt.plot(t, vel) 

plt.subplot(3,1,3)
plt.plot(t, acc) 

fig = plt.figure()
ax1 = p3.Axes3D(fig)
line, = ax1.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'b.')

ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_title('Trajectory')
ax1.view_init(50, 50)
plt.show()
