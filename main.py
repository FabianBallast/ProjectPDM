import gym, env
import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
from obstacles.ObstacleHandler import ObstacleHandler
from obstacles.Obstacle import Shelf
from obstacles.Obstacle import Forklift
from path_planning.RRT import RRT
from path_planning.RRTstar import RRTstar
from util.traj_from_line import point_from_traj
import geom_controller as cont 
from path_planning.TrajectoryOptimization import find_kinodynamic_trajectory, create_trajectory_from_vertices
from path_planning.Trajectory import plot_trajectories

### Start and goal
endOfTime = 30
fast_obst = True # Options: True and False. 
kinodynamic = False # Use kinodynamic RRT* extention (use with RRTstar)
environment = 2 # 0, 1 or 2, for different environments
# Each environment has one moving forklift and a number of static shelves

obs_list = []
if(environment == 0):
    start = np.array([9,9,1,0])
    goal = np.array([1,1,9,endOfTime])
    ### Static obstacles
    # First array indicates the positon, second array the size of the shelf
    sob1 = Shelf(np.array([3,4,5,endOfTime/2]), np.array([2,8,10,endOfTime])) # Added mid_time and max_time

    # Another shelf
    # First array indicates the positon, second array the size of the shelf
    sob2 = Shelf(np.array([7,6,5,endOfTime/2]), np.array([2,8,10,endOfTime])) # Added mid_time and max_time

    sobs_list = [sob1, sob2]

    ### Add Dynamic obstacles, can be expressed as static obstacles in the xyzt space,
    if fast_obst:
        state_begin = np.array([5,9,4])
        state_end = np.array([5,1,4])
        time_begin = 5
        time_end = 20
    else:
        state_begin = np.array([5,9,4])
        state_end = np.array([5,7,4])
        time_begin = 11
        time_end = 15
    block_size = np.array([1,2,8])

elif(environment == 1):
    start = np.array([9,9,1,0])
    goal = np.array([1,1,9,endOfTime])
    ### Static obstacles
    # First array indicates the positon, second array the size of the shelf
    sob1 = Shelf(np.array([3,4,5,endOfTime/2]), np.array([2,8,10,endOfTime])) # Added start_time and max_time

    # Shelf with hole, made up of 4 seperate shelfs
    # First array indicates the positon, second array the size of the shelf
    sob2 = Shelf(np.array([7,6,1.5,endOfTime/2]), np.array([2,8,3,endOfTime])) # Added mid_time and max_time
    sob3 = Shelf(np.array([7,6,8.5,endOfTime/2]), np.array([2,8,3,endOfTime])) # Added mid_time and max_time
    sob4 = Shelf(np.array([7,8.5,5,endOfTime/2]), np.array([2,3,4,endOfTime])) # Added mid_time and max_time
    sob5 = Shelf(np.array([7,3.5,5,endOfTime/2]), np.array([2,3,4,endOfTime])) # Added mid_time and max_time

    sobs_list = [sob1, sob2, sob3, sob4, sob5]

    ### Add Dynamic obstacles, can be expressed as static obstacles in the xyzt space,
    if fast_obst:
        state_begin = np.array([5,9,4])
        state_end = np.array([5,1,4])
        time_begin = 5
        time_end = 20
    else:
        state_begin = np.array([5,9,4])
        state_end = np.array([5,7,4])
        time_begin = 11
        time_end = 15
    block_size = np.array([1,2,8])

elif(environment == 2):
    start = np.array([9,1,1,0])
    goal = np.array([5,3,1,endOfTime])
    ### Static obstacles
    # First array indicates the positon, second array the size of the shelf
    sob1 = Shelf(np.array([3,5,5,endOfTime/2]), np.array([2,6,10,endOfTime])) # Added start_time and max_time

    # Shelf with hole, made up of 4 seperate shelfs
    # First array indicates the positon, second array the size of the shelf
    sob2 = Shelf(np.array([7,6,1.5,endOfTime/2]), np.array([2,8,3,endOfTime])) # Added mid_time and max_time
    sob3 = Shelf(np.array([7,6,8.5,endOfTime/2]), np.array([2,8,3,endOfTime])) # Added mid_time and max_time
    sob4 = Shelf(np.array([7,8.5,5,endOfTime/2]), np.array([2,3,4,endOfTime])) # Added mid_time and max_time
    sob5 = Shelf(np.array([7,3.5,5,endOfTime/2]), np.array([2,3,4,endOfTime])) # Added mid_time and max_time

    sobs_list = [sob1, sob2, sob3, sob4, sob5]

    ### Add Dynamic obstacles, can be expressed as static obstacles in the xyzt space,
    if fast_obst:
        state_begin = np.array([5,9,4])
        state_end = np.array([5,1,4])
        time_begin = 1
        time_end = 10
    else:
        state_begin = np.array([5,9,4])
        state_end = np.array([5,7,4])
        time_begin = 11
        time_end = 15
    block_size = np.array([1,2,8])

# Block before movement
dob1 = Forklift(np.append(state_begin,time_begin/2), np.append(block_size,time_begin)) 
# Block after movement
dob2 = Forklift(np.append(state_end,(endOfTime + time_end)/2), np.append(block_size,endOfTime - time_end))
# Block during movement
dobs_list = [dob1, dob2]
# find minimum number of blocks for approximation
vector_to_travel = state_end-state_begin
time_to_travel = time_end - time_begin
velocity_vector = np.abs(vector_to_travel)/time_to_travel
time_per_block = np.max(block_size[velocity_vector!= 0]/velocity_vector[velocity_vector!= 0])
minimum_blocks = np.ceil(np.max(np.abs(vector_to_travel)/block_size))
approximation_range = int(minimum_blocks*2)
for i in range(approximation_range):
    state = state_begin + (i/approximation_range)*vector_to_travel
    time  = time_begin + (i/approximation_range)*time_to_travel
    ob = Forklift(np.append(state,time), np.append(block_size,time_per_block))
    dobs_list.append(ob)

# Sort the dynamic obstacles by time
dobs_list.sort(key = lambda obs : obs.position[3])

# Save all obstacles in ObstacleHandler
obHand = ObstacleHandler(sobs_list + dobs_list)

### Pick one of the path planning methods
# The first array indicates the max configuration space, the second represents the obstacles
# path = RRT(np.array([10, 10, 10, endOfTime]), obHand)
path = RRTstar(np.array([10, 10, 10, endOfTime]), obHand)

tree = path.find_path(start, goal, 500)
print("Path found!")
curr_goal_ind = 1
curr_goal = tree.sorted_vertices[curr_goal_ind].state
past_goal = tree.sorted_vertices[curr_goal_ind - 1].state
t0 = 0
timeToNode = curr_goal[3] - past_goal[3]  # Time is now determined by the state and no longer constant

env = gym.make('Quadrotor-v0')
current_state = env.reset(position=start[0:3])
controller = cont.controlller()


if(kinodynamic):
    ## Create optimized trajectories
    curr_path_index = 0
    t0 = 0
    smooth_path, t_traj = create_trajectory_from_vertices(tree.sorted_vertices)
    t_end = t_traj[1]
    dt = 0.01
    t = 0

    real_trajectory = {'x': [], 'y': [], 'z': []}
    des_trajectory = {'x': [], 'y': [], 'z': []}

    final_goal_reached = False
    while not final_goal_reached:
        trajec_x = smooth_path[0][curr_path_index].evaluate_track(t)
        trajec_y = smooth_path[1][curr_path_index].evaluate_track(t)
        trajec_z = smooth_path[2][curr_path_index].evaluate_track(t)

        # Added the [0:3] to only take x, y and z
        trajectory_goal = [np.asarray([trajec_x[0], trajec_y[0], trajec_z[0]]), \
                        np.asarray([trajec_x[1], trajec_y[1], trajec_z[1]]), \
                        np.asarray([trajec_x[2], trajec_y[2], trajec_z[2]]), 0, 0]
        control_input = controller.control(trajectory_goal, current_state)
        obs, reward, done, info = env.step(control_input['cmd_motor_speeds'])

        real_trajectory['x'].append(obs['x'][0])
        real_trajectory['y'].append(obs['x'][1])
        real_trajectory['z'].append(obs['x'][2])

        des_trajectory['x'].append(trajectory_goal[0][0])
        des_trajectory['y'].append(trajectory_goal[0][1])
        des_trajectory['z'].append(trajectory_goal[0][2])
        current_state = obs
        # Update time 
        t += dt
        # Check if goal is reached
        if np.linalg.norm(np.array([obs['x'][0], obs['x'][1], obs['x'][2]]) - curr_goal[0:3]) < 0.01 and np.all(curr_goal == goal):
            print("Done!")
            final_goal_reached = True
            break
        # Check if close enough to current_goal to change direction to new goal
        elif np.linalg.norm(np.array([obs['x'][0], obs['x'][1], obs['x'][2]]) - curr_goal[0:3]) < 0.25 and not np.all(curr_goal == goal):
            curr_goal_ind += 1
            past_goal = curr_goal
            curr_goal = tree.sorted_vertices[curr_goal_ind].state
            timeToNode = curr_goal[3] - past_goal[3]
            t0 = t
        elif t > t_traj[-1]:
            print("Reached goal!")
            break
        elif t > t_end:
            curr_path_index += 1
            t_end = t_traj[curr_path_index+1]
        elif(t>endOfTime):
            print("Beyond simulation time reached, trajectory following failed")
            break
else:
    dt_fraction = 400
    dt = timeToNode/dt_fraction #Variable, dt is now determined by fraction of the timeToNode
    t = 0

    real_trajectory = {'x': [], 'y': [], 'z': []}
    des_trajectory = {'x': [], 'y': [], 'z': []}

    final_goal_reached = False
    while not final_goal_reached:
        trajec = point_from_traj(past_goal[0:3], curr_goal[0:3], t0+timeToNode, t, t0)
        # Added the [0:3] to only take x, y and z
        trajectory_goal = [trajec[0][0:3], trajec[1][0:3], trajec[2][0:3], 0, 0]
        control_input = controller.control(trajectory_goal, current_state)
        obs, reward, done, info = env.step(control_input['cmd_motor_speeds'])

        real_trajectory['x'].append(obs['x'][0])
        real_trajectory['y'].append(obs['x'][1])
        real_trajectory['z'].append(obs['x'][2])

        des_trajectory['x'].append(trajectory_goal[0][0])
        des_trajectory['y'].append(trajectory_goal[0][1])
        des_trajectory['z'].append(trajectory_goal[0][2])
        current_state = obs
        # Update time 
        t += dt
        # Check if goal is reached
        if np.linalg.norm(np.array([obs['x'][0], obs['x'][1], obs['x'][2]]) - curr_goal[0:3]) < 0.01 and np.all(curr_goal == goal):
            print("Done!")
            final_goal_reached = True
            break
        # Check if close enough to current_goal to change direction to new goal
        elif np.linalg.norm(np.array([obs['x'][0], obs['x'][1], obs['x'][2]]) - curr_goal[0:3]) < 0.25 and not np.all(curr_goal == goal):
            curr_goal_ind += 1
            past_goal = curr_goal
            # print(curr_goal)
            curr_goal = tree.sorted_vertices[curr_goal_ind].state
            timeToNode = curr_goal[3] - past_goal[3]
            dt = timeToNode/dt_fraction #Variable, dt is determined by fraction of the timeToNode
            t0 = t
        elif(t>endOfTime):
            raise ValueError("Beyond simulation time reached, trajectory following failed")

#%%
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
fork = ax1.add_collection3d(dobs_list[0].get_plot_obstacle())
line, = ax1.plot([real_trajectory['x'][0]], [real_trajectory['y'][0]], [real_trajectory['z'][0]], label='Real_Trajectory')
lineRef, = ax1.plot([des_trajectory['x'][0]], [des_trajectory['y'][0]], [des_trajectory['z'][0]], label='Desired_Trajectory')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_xlim3d(0, 10)
ax1.set_ylim3d(0, 10)
ax1.set_zlim3d(0, 10)
ax1.set_title('3D animate')
# ax1.view_init(20, -16)
ax1.view_init(90, 90)
# ax1.view_init(44, -20)

# Plot obstacles to test placement
obHand.plot_obstacles(ax1)

tree.plot_tree(ax1)
ax1.legend(loc='lower right')

# Animate moving objects
anim_len = len(real_trajectory['x'])
# Forklift moves in straight line, detailed manual path gen for smooth animation
def forklift_path(i):
    anim_percent = i/anim_len # Proportion of animation completed
    sim_i = anim_percent*endOfTime # Simulated time
    if(sim_i < dob1.position[3]):
        return dob1.position[0:3]
    elif (sim_i < dob2.position[3]):
        return dob1.position[0:3] + (sim_i - dob1.position[3])*(dob2.position[0:3]-dob1.position[0:3])/ \
            (dob2.position[3]-dob1.position[3])
    else:
        return dob2.position[0:3]

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
    fork_pos = forklift_path(i)
    fork_temp = Forklift(np.append(fork_pos,0), np.append(block_size,0))
    fork.set_verts(fork_temp.get_obs_verts())
    try:
        fork.do_3d_projection() # Results in a warning, but does not work without this
    except:
        pass # Just ignore the warning without log
    
    return line, lineRef, point, fork

ani = animation.FuncAnimation(fig=fig,
                              func=animate,
                              frames=anim_len,
                              interval=dt*100,
                              repeat=False,
                              blit=True)

plt.show()

# print("Saving animation ... (takes a minute or so)")
# f = r"./animation.gif" 
# writergif = animation.PillowWriter(fps=30) 
# ani.save(f, writer=writergif)
# print("Saved animation!")
