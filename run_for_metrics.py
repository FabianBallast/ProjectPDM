import gym, env
import numpy as np
import numpy.random as rand
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
iterations = 1 # Run RRT and RRTstar this many times each for each env

# Environment 0
print("Environment 0")
obs_list = []
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

# Run RRT and RRTstar a bunch of times to compare
sum_RRTstar_dist0 = 0
sum_RRT_dist0 = 0
sum_RRTstar_time0 = 0
sum_RRT_time0 = 0
sum_kino_RRT0 = 0
kino_fails0 = 0

for i in range(iterations):
    # The first array indicates the max configuration space, the second represents the obstacles
    path = RRTstar(np.array([10, 10, 10, endOfTime]), obHand, rand.randint(0,100))
    print(f"RRTstar iteration: {i+1}")

    tree = path.find_path(start, goal, 500)
    print("Path found!")
    root_node = tree.sorted_vertices[0]
    goal_node = tree.sorted_vertices[-1]
    smooth_path, t_traj = create_trajectory_from_vertices(tree.sorted_vertices,optimize=False)
    sum_RRTstar_time0 += t_traj[-1]
    try:
        smooth_path, t_traj = create_trajectory_from_vertices(tree.sorted_vertices,optimize=True)
        sum_kino_RRT0 += t_traj[-1]
    except:
        print("Kino path failed, disregarded")
        kino_fails0 += 1
        pass
    sum_RRTstar_dist0 += goal_node.distance_to_root()


for i in range(iterations):
    # The first array indicates the max configuration space, the second represents the obstacles
    path = RRT(np.array([10, 10, 10, endOfTime]), obHand, rand.randint(0,100))
    print(f"RRT iteration: {i+1}")

    tree = path.find_path(start, goal, 500)
    print("Path found!")
    root_node = tree.sorted_vertices[0]
    goal_node = tree.sorted_vertices[-1]
    smooth_path, t_traj = create_trajectory_from_vertices(tree.sorted_vertices,optimize=False)
    sum_RRT_time0 += t_traj[-1]
    sum_RRT_dist0 += goal_node.distance_to_root()

if(iterations-kino_fails0 != 0):
    print("avg_kinodynamic_RRTstar_time", sum_kino_RRT0/(iterations-kino_fails0))
else:
    print("Kino failed every time")
print("avg_RRTstar_time", sum_RRTstar_time0/iterations)
print("avg_RRT_time", sum_RRT_time0/iterations)
print("avg_RRTstar_dist", sum_RRTstar_dist0/iterations)
print("avg_RRT_dist", sum_RRT_dist0/iterations)

# Environment 1
print("Environment 1")
obs_list = []
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

# Run RRT and RRTstar a bunch of times to compare
sum_RRTstar_dist1 = 0
sum_RRT_dist1 = 0
sum_RRTstar_time1 = 0
sum_RRT_time1 = 0
sum_kino_RRT1 = 0
kino_fails1 = 0

for i in range(iterations):
    # The first array indicates the max configuration space, the second represents the obstacles
    path = RRTstar(np.array([10, 10, 10, endOfTime]), obHand, rand.randint(0,100))
    print(f"RRTstar iteration: {i+1}")

    tree = path.find_path(start, goal, 500)
    print("Path found!")
    root_node = tree.sorted_vertices[0]
    goal_node = tree.sorted_vertices[-1]
    smooth_path, t_traj = create_trajectory_from_vertices(tree.sorted_vertices,optimize=False)
    sum_RRTstar_time1 += t_traj[-1]
    try:
        smooth_path, t_traj = create_trajectory_from_vertices(tree.sorted_vertices,optimize=True)
        sum_kino_RRT1 += t_traj[-1]
    except:
        print("Kino path failed, disregarded")
        kino_fails0 += 1
        pass
    sum_RRTstar_dist1 += goal_node.distance_to_root()


for i in range(iterations):
    # The first array indicates the max configuration space, the second represents the obstacles
    path = RRT(np.array([10, 10, 10, endOfTime]), obHand, rand.randint(0,100))
    print(f"RRT iteration: {i+1}")

    tree = path.find_path(start, goal, 500)
    print("Path found!")
    root_node = tree.sorted_vertices[0]
    goal_node = tree.sorted_vertices[-1]
    smooth_path, t_traj = create_trajectory_from_vertices(tree.sorted_vertices,optimize=False)
    sum_RRT_time1 += t_traj[-1]
    sum_RRT_dist1 += goal_node.distance_to_root()

if(iterations-kino_fails1 != 0):
    print("avg_kinodynamic_RRTstar_time", sum_kino_RRT1/(iterations-kino_fails1))
else:
    print("Kino failed every time")
print("avg_RRTstar_time", sum_RRTstar_time1/iterations)
print("avg_RRT_time", sum_RRT_time1/iterations)
print("avg_RRTstar_dist", sum_RRTstar_dist1/iterations)
print("avg_RRT_dist", sum_RRT_dist1/iterations)

# Environment 2
print("Environment 2")
obs_list = []
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

# Run RRT and RRTstar a bunch of times to compare
sum_RRTstar_dist2 = 0
sum_RRT_dist2 = 0
sum_RRTstar_time2 = 0
sum_RRT_time2 = 0
sum_kino_RRT2 = 0
kino_fails2 = 0

for i in range(iterations):
    # The first array indicates the max configuration space, the second represents the obstacles
    path = RRTstar(np.array([10, 10, 10, endOfTime]), obHand, rand.randint(0,100))
    print(f"RRTstar iteration: {i+1}")

    tree = path.find_path(start, goal, 500)
    print("Path found!")
    root_node = tree.sorted_vertices[0]
    goal_node = tree.sorted_vertices[-1]
    smooth_path, t_traj = create_trajectory_from_vertices(tree.sorted_vertices,optimize=False)
    sum_RRTstar_time2 += t_traj[-1]
    try:
        smooth_path, t_traj = create_trajectory_from_vertices(tree.sorted_vertices,optimize=True)
        sum_kino_RRT2 += t_traj[-1]
    except:
        print("Kino path failed, disregarded")
        kino_fails2+=1
        pass
    sum_RRTstar_dist2 += goal_node.distance_to_root()


for i in range(iterations):
    # The first array indicates the max configuration space, the second represents the obstacles
    path = RRT(np.array([10, 10, 10, endOfTime]), obHand, rand.randint(0,100))
    print(f"RRT iteration: {i+1}")

    tree = path.find_path(start, goal, 500)
    print("Path found!")
    root_node = tree.sorted_vertices[0]
    goal_node = tree.sorted_vertices[-1]
    smooth_path, t_traj = create_trajectory_from_vertices(tree.sorted_vertices,optimize=False)
    sum_RRT_time2 += t_traj[-1]
    sum_RRT_dist2 += goal_node.distance_to_root()

if(iterations-kino_fails2 != 0):
    print("avg_kinodynamic_RRTstar_time", sum_kino_RRT2/(iterations-kino_fails2))
else:
    print("Kino failed every time")
print("avg_RRTstar_time", sum_RRTstar_time2/iterations)
print("avg_RRT_time", sum_RRT_time2/iterations)
print("avg_RRTstar_dist", sum_RRTstar_dist2/iterations)
print("avg_RRT_dist", sum_RRT_dist2/iterations)


# Total performance
print("All environments")
print("total avg_kinodynamic_RRTstar_time", (sum_kino_RRT0+sum_kino_RRT1+sum_kino_RRT2)/3)
print("total avg_RRTstar_time", (sum_RRTstar_time0+sum_RRTstar_time1+sum_RRTstar_time2)/3)
print("total avg_RRT_time", (sum_RRT_time0+sum_RRT_time1+sum_RRT_time2)/3)
print("total avg_RRTstar_dist", (sum_RRTstar_dist0+sum_RRTstar_dist1+sum_RRTstar_dist2)/3)
print("total avg_RRT_dist", (sum_RRT_dist0+sum_RRT_dist1+sum_RRT_dist2)/3)