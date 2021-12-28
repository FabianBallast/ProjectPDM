import gym
from numpy.linalg.linalg import norm
import env
import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
from obstacles.ObstacleHandler import ObstacleHandler
from obstacles.Obstacle import Shelf


ob1 = Shelf(np.array([2,2,4]), np.array([4,4,8]))
obHand = ObstacleHandler([ob1])


### Pick one
import geom_controller as cont # Best performing
# import PD_controller as cont # Works on simple trajectories, but tud is too fast

### Pick one
# from util.hover import hover as traj
# from util.circle import circle as traj
# from util.diamond import diamond as traj
from util.tud import tud as traj

env = gym.make('Quadrotor-v0')
start = traj(0)[0]
current_state = env.reset(position=start)
controller = cont.controlller()

# print("current:", current_state)
dt = 0.01
t = 0

track_error = 0
bat_usage = 0
T_fin = 0

real_trajectory = {'x': [], 'y': [], 'z': []}
des_trajectory = {'x': [], 'y': [], 'z': []}
while(t < 25):
    trajectory_goal = traj(t)
    # print(trajectory_goal[0])
    # print(current_state['x'])
    # print(f"Time {t}")
    control_input = controller.control(trajectory_goal, current_state)
    # print(f"Input {control_input['cmd_motor_speeds']}")
    # print(f"Control force {control_input['cmd_thrust']}")
    # print(f"Control momnt {control_input['cmd_moment']}")
    obs, reward, done, info = env.step(control_input['cmd_motor_speeds'])
    # print(obs['x'])
    # print(obs['v'])
    # print()

    track_error= track_error + norm(current_state['x'] - trajectory_goal[0])**2 * dt
    bat_usage=bat_usage + np.sum(control_input['cmd_motor_speeds']**2)* dt

    real_trajectory['x'].append(obs['x'][0])
    real_trajectory['y'].append(obs['x'][1])
    real_trajectory['z'].append(obs['x'][2])

    des_trajectory['x'].append(trajectory_goal[0][0])
    des_trajectory['y'].append(trajectory_goal[0][1])
    des_trajectory['z'].append(trajectory_goal[0][2])
    current_state = obs

    if np.all(trajectory_goal[5] == True):
        T_fin = t
        break
    else:
        t += dt

    
print(f"Tracking error: {track_error}")
print(f"Time to finish task: {T_fin}")
print(f"Battery usage: {bat_usage}")


fig = plt.figure()
ax1 = p3.Axes3D(fig, auto_add_to_figure=False) # 3D place for drawing
fig.add_axes(ax1)
real_trajectory['x'] = np.array(real_trajectory['x'])
real_trajectory['y'] = np.array(real_trajectory['y'])
real_trajectory['z'] = np.array(real_trajectory['z'])
des_trajectory['x'] = np.array(des_trajectory['x'])
des_trajectory['y'] = np.array(des_trajectory['y'])
des_trajectory['z'] = np.array(des_trajectory['z'])
point, = ax1.plot([real_trajectory['x'][0]], [real_trajectory['y'][0]], [real_trajectory['z'][0]], 'ro', label='Quadrotor')
line, = ax1.plot([real_trajectory['x'][0]], [real_trajectory['y'][0]], [real_trajectory['z'][0]], label='Real_Trajectory')
lineRef, = ax1.plot([des_trajectory['x'][0]], [des_trajectory['y'][0]], [des_trajectory['z'][0]], label='Desired_Trajectory')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_xlim3d(0, 10)
ax1.set_ylim3d(0, 10)
ax1.set_zlim3d(0, 10)
ax1.set_title('3D animate')
ax1.view_init(0, 0)
ax1.legend(loc='lower right')
obHand.plot_obstacles(ax1)

def animate(i):
    line.set_xdata(real_trajectory['x'][:i + 1])
    line.set_ydata(real_trajectory['y'][:i + 1])
    line.set_3d_properties(real_trajectory['z'][:i + 1])
    lineRef.set_xdata(des_trajectory['x'][:i + 1])
    lineRef.set_ydata(des_trajectory['y'][:i + 1])
    lineRef.set_3d_properties(des_trajectory['z'][:i + 1])
    point.set_xdata([real_trajectory['x'][i]])
    point.set_ydata([real_trajectory['y'][i]])
    point.set_3d_properties([real_trajectory['z'][i]])

ani = animation.FuncAnimation(fig=fig,
                              func=animate,
                              frames=len(real_trajectory['x']),
                              interval=1,
                              repeat=False,
                              blit=False)
plt.show()