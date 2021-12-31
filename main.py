import gym, env
import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
from obstacles.ObstacleHandler import ObstacleHandler
from obstacles.Obstacle import Shelf
from path_planning.RRT import RRT
from path_planning.RRTstar import RRTstar
from util.traj_from_line import point_from_traj
import geom_controller as cont # Best performing

### Start and goal
start = np.array([9,9,1])
goal = np.array([1,1,9])

### Temporary obstacles
ob1 = Shelf(np.array([3,4,5]), np.array([2,8,10]))
# ob2 = Shelf(np.array([7,6,5]), np.array([2,8,10])) Simple shelf

# Shelf with hole
ob2 = Shelf(np.array([7,6,1.5]), np.array([2,8,3]))
ob3 = Shelf(np.array([7,6,8.5]), np.array([2,8,3]))
ob4 = Shelf(np.array([7,8.5,5]), np.array([2,3,4]))
ob5 = Shelf(np.array([7,3.5,5]), np.array([2,3,4]))
obHand = ObstacleHandler([ob1, ob2, ob3, ob4, ob5])

# ### Grid for obstacle detection test
# x, y, z = np.meshgrid(np.linspace(0, 10, 6), np.linspace(0, 10, 6), np.linspace(0, 10, 6))

### Pick one
# path = RRT(np.array([10, 10, 10]), obHand)
path = RRTstar(np.array([10, 10, 10]), obHand)

tree = path.find_path(start, goal, 200)
curr_goal_ind = 1
curr_goal = tree.sorted_vertices[curr_goal_ind].state
past_goal = tree.sorted_vertices[curr_goal_ind - 1].state
t0 = 0
timeToNode = 5

env = gym.make('Quadrotor-v0')
current_state = env.reset(position=start)
controller = cont.controlller()

# print("current:", current_state)
dt = 0.01
t = 0

real_trajectory = {'x': [], 'y': [], 'z': []}
des_trajectory = {'x': [], 'y': [], 'z': []}

final_goal_reached = False
while not final_goal_reached:
    trajec = point_from_traj(past_goal, curr_goal, t0+timeToNode, t, t0)
    trajectory_goal = [trajec[0], trajec[1], trajec[2], 0, 0]
    control_input = controller.control(trajectory_goal, current_state)
    obs, reward, done, info = env.step(control_input['cmd_motor_speeds'])

    real_trajectory['x'].append(obs['x'][0])
    real_trajectory['y'].append(obs['x'][1])
    real_trajectory['z'].append(obs['x'][2])

    des_trajectory['x'].append(trajectory_goal[0][0])
    des_trajectory['y'].append(trajectory_goal[0][1])
    des_trajectory['z'].append(trajectory_goal[0][2])
    current_state = obs

    t += dt

    if np.linalg.norm(np.array([obs['x'][0], obs['x'][1], obs['x'][2]]) - curr_goal) < 0.01 and np.all(curr_goal == goal):
        print("Done!")
        final_goal_reached = True
        break
    elif np.linalg.norm(np.array([obs['x'][0], obs['x'][1], obs['x'][2]]) - curr_goal) < 0.25 and not np.all(curr_goal == goal):
        curr_goal_ind += 1
        past_goal = curr_goal
        curr_goal = tree.sorted_vertices[curr_goal_ind].state
        t0 = t


fig = plt.figure()
ax1 = p3.Axes3D(fig, auto_add_to_figure=False) # 3D place for drawing
fig.add_axes(ax1)
real_trajectory['x'] = np.array(real_trajectory['x'])
real_trajectory['y'] = np.array(real_trajectory['y'])
real_trajectory['z'] = np.array(real_trajectory['z'])
des_trajectory['x'] = np.array(des_trajectory['x'])
des_trajectory['y'] = np.array(des_trajectory['y'])
des_trajectory['z'] = np.array(des_trajectory['z'])
point, = ax1.plot([real_trajectory['x'][0]], [real_trajectory['y'][0]], [real_trajectory['z'][0]], 'co', label='Quadrotor')
line, = ax1.plot([real_trajectory['x'][0]], [real_trajectory['y'][0]], [real_trajectory['z'][0]], label='Real_Trajectory')
lineRef, = ax1.plot([des_trajectory['x'][0]], [des_trajectory['y'][0]], [des_trajectory['z'][0]], label='Desired_Trajectory')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_xlim3d(0, 10)
ax1.set_ylim3d(0, 10)
ax1.set_zlim3d(0, 10)
ax1.set_title('3D animate')
ax1.view_init(40, 40)

# Plot obstacles to test placement
obHand.plot_obstacles(ax1)

# # Plot points in grid to test obstacle detection
# for point in list(zip(x.flatten(), y.flatten(), z.flatten())):
#     if obHand.point_in_obstacle(point):
#         ax1.plot([point[0]], [point[1]], [point[2]], 'bo')
#     else:
#         ax1.plot([point[0]], [point[1]], [point[2]], 'go')


tree.plot_tree(ax1)
ax1.legend(loc='lower right')

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

    return line, lineRef, point

ani = animation.FuncAnimation(fig=fig,
                              func=animate,
                              frames=len(real_trajectory['x']),
                              interval=dt*1000,
                              repeat=False,
                              blit=True)

plt.show()