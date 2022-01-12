import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace

try:
    from path_planning.Trajectory import smax, jmax, amax, vmax, crit_value
    from path_planning.Trajectory import find_xC, Trajectory, plot_trajectories
except ModuleNotFoundError:
    from Trajectory import smax, jmax, amax, vmax, crit_value
    from Trajectory import find_xC, Trajectory, plot_trajectories

def find_kinodynamic_trajectory(start, end, t0=0, t_end=None):
    """
    Find a suboptimal kinodynamic trajectory from start to end.

    Args:
        - Start: Numpy array of shape (3x3) as [[x y z], [dx dy dz], [ddx ddy ddz]] of start state
        - End:   Numpy array of shape (3x3) as [[x y z], [dx dy dz], [ddx ddy ddz]] of end state
        - t0: Starting time of trajectory.
        - t_end: Required stop time of trajectory. If None, then automatically the fastest feasible trajectory is generated.
    
    Returns:
        - Trajectories: List of 3 trajectories, one for x, y and z each.
    """
    T_x, T_arr_x, vD_arr_x, vD_arr_B_x, vD_arr_G_x, aB_arr_x, aG_arr_x, x_traj = find_shortest_time(start[0,0], end[0,0], start[1,0], end[1,0], start[2,0], end[2,0], t0)
    T_y, T_arr_y, vD_arr_y, vD_arr_B_y, vD_arr_G_y, aB_arr_y, aG_arr_y, y_traj = find_shortest_time(start[0,1], end[0,1], start[1,1], end[1,1], start[2,1], end[2,1], t0)
    T_z, T_arr_z, vD_arr_z, vD_arr_B_z, vD_arr_G_z, aB_arr_z, aG_arr_z, z_traj = find_shortest_time(start[0,2], end[0,2], start[1,2], end[1,2], start[2,2], end[2,2], t0)

    if t_end is None:
        t_end = max([T_x, T_y, T_z])

    if T_x == t_end:
        y_traj = __synchronize(T_x, T_arr_y, vD_arr_y, vD_arr_B_y, vD_arr_G_y, aB_arr_y, aG_arr_y, start[:, 1], end[:, 1], t0)
        z_traj = __synchronize(T_x, T_arr_z, vD_arr_z, vD_arr_B_z, vD_arr_G_z, aB_arr_z, aG_arr_z, start[:, 2], end[:, 2], t0)
    elif T_y == t_end:
        x_traj = __synchronize(T_y, T_arr_x, vD_arr_x, vD_arr_B_x, vD_arr_G_x, aB_arr_x, aG_arr_x, start[:, 0], end[:, 0], t0)
        z_traj = __synchronize(T_y, T_arr_z, vD_arr_z, vD_arr_B_z, vD_arr_G_z, aB_arr_z, aG_arr_z, start[:, 2], end[:, 2], t0)
    elif T_z == t_end:
        x_traj = __synchronize(T_z, T_arr_x, vD_arr_x, vD_arr_B_x, vD_arr_G_x, aB_arr_x, aG_arr_x, start[:, 0], end[:, 0], t0)
        y_traj = __synchronize(T_z, T_arr_y, vD_arr_y, vD_arr_B_y, vD_arr_G_y, aB_arr_y, aG_arr_y, start[:, 1], end[:, 1], t0)
    else:
        x_traj = __synchronize(t_end, T_arr_x, vD_arr_x, vD_arr_B_x, vD_arr_G_x, aB_arr_x, aG_arr_x, start[:, 0], end[:, 0], t0)
        y_traj = __synchronize(t_end, T_arr_y, vD_arr_y, vD_arr_B_y, vD_arr_G_y, aB_arr_y, aG_arr_y, start[:, 1], end[:, 1], t0)
        z_traj = __synchronize(t_end, T_arr_z, vD_arr_z, vD_arr_B_z, vD_arr_G_z, aB_arr_z, aG_arr_z, start[:, 2], end[:, 2], t0)
  
    return x_traj, y_traj, z_traj

def __synchronize(Tmax, T_arr, vD_arr, vD_arr_B, vD_arr_G, aB_arr, aG_arr, start, end, t0):
    """
    Synchronize the trajectories such that they all start at t0 and end after 'end' seconds.
    """
    if np.max(T_arr) < Tmax:
        raise Exception("Cannot sync these trajectories.")

    # plt.figure()
    # plt.plot(vD_arr,T_arr)

    if T_arr[0] > T_arr[-1]:
        vD_sync = np.interp(Tmax, np.flip(T_arr), np.flip(vD_arr))
    else:
        vD_sync = np.interp(Tmax, T_arr, vD_arr)

    if vD_arr_B[0] > vD_arr_B[-1]:
        vD_arr_B = np.flip(vD_arr_B)
        aB_arr = np.flip(aB_arr)

    if vD_arr_G[0] > vD_arr_G[-1]:
        vD_arr_G = np.flip(vD_arr_G)
        aG_arr = np.flip(aG_arr)

    aB = np.interp(vD_sync, vD_arr_B, aB_arr)
    aG = np.interp(vD_sync, vD_arr_G, aG_arr)

    _, vD_tB0_b = find_aB_vD(start[1], start[2], np.array([aB]), start[0]>end[0])
    _, vD_tB0_g = find_aB_vD(end[1], -end[2], np.array([aG]), start[0]>end[0])

    tA1, tA2, tC1, tC2, tE1, tE2, tH1, tH2, tB, tG = find_times(np.array([aB]), np.array([vD_sync]), start[2], np.array([aG]), np.array([vD_sync]), -end[2], vD_tB0_b, vD_tB0_g)
    xC = find_xC(tA1, tA2, tC1, tC2, tB, start[0], start[1], start[2], aB)
    xE = 2*end[0] - find_xC(tH1, tH2, tE1, tE2, tG, end[0], end[1], -end[2], aG)
    delta_X = xE - xC
    
    time_stamps = { 'tA1': tA1[0], 'tA2': tA2[0], 'tC1': tC1[0], 'tC2': tC2[0], 'tB' : tB[0], 'tD' : abs(delta_X[-1]) / abs(vD_sync), 
                    'tE1': tE1[0], 'tE2': tE2[0], 'tH1': tH1[0], 'tH2': tH2[0], 'tG' : tG[0]}

    return Trajectory(start, end, time_stamps, aB, aG, t0)


def find_aB_vD(v0, a0, aB_arr):

    vD_arr = np.zeros_like(aB_arr)

    case_a = (abs(aB_arr - a0) >  crit_value) & (abs(aB_arr) >  crit_value) & (aB_arr >=  abs(a0))
    case_b = (abs(aB_arr - a0) >  crit_value) & (abs(aB_arr) <= crit_value) & (aB_arr >=  abs(a0))
    case_c = (abs(aB_arr - a0) <= crit_value) & (abs(aB_arr) >  crit_value) & (aB_arr >=  abs(a0))
    case_d = (abs(aB_arr - a0) <= crit_value) & (abs(aB_arr) <= crit_value) & (aB_arr >=  abs(a0))

    case_e = (abs(aB_arr - a0) >  crit_value) & (abs(aB_arr) >  crit_value) & (aB_arr <=  -abs(a0))
    case_f = (abs(aB_arr - a0) >  crit_value) & (abs(aB_arr) <= crit_value) & (aB_arr <=  -abs(a0))
    case_g = (abs(aB_arr - a0) <= crit_value) & (abs(aB_arr) >  crit_value) & (aB_arr <=  -abs(a0))
    case_h = (abs(aB_arr - a0) <= crit_value) & (abs(aB_arr) <= crit_value) & (aB_arr <=  -abs(a0))

    case_i = (aB_arr < abs(a0)) & (aB_arr >= 0) 
    case_j = (aB_arr > -abs(a0)) & (aB_arr < 0)     
    
    vD_arr[case_a] = v0 + (2*aB_arr[case_a]**2 - a0**2) / (2*jmax) + (jmax * (a0 + 2*aB_arr[case_a]) / (2*smax))
    vD_arr[case_b] = v0 + (jmax * (a0 + aB_arr[case_b]) +  2 * aB_arr[case_b]*np.sqrt(smax * aB_arr[case_b])) / (2*smax) - (a0**2 - aB_arr[case_b]**2) / (2*jmax)
    vD_arr[case_c] = v0 + aB_arr[case_c]**2 / (2 * jmax) + (a0 + aB_arr[case_c]) * np.sqrt((aB_arr[case_c]-a0) / smax) + aB_arr[case_c] * jmax / (2*smax)
    vD_arr[case_d] = v0 + (2*(a0 + aB_arr[case_d])*np.sqrt(aB_arr[case_d]-a0)+2*aB_arr[case_d]*np.sqrt(aB_arr[case_d])) / (2*np.sqrt(smax))
    vD_arr[case_e] = v0 - (2*aB_arr[case_e]**2 - a0**2) / (2*jmax) + (jmax * (a0 + 2*aB_arr[case_e]) / (2*smax))
    vD_arr[case_f] = v0 + (jmax * (a0 + aB_arr[case_f]) +  2 * aB_arr[case_f]*np.sqrt(-smax * aB_arr[case_f])) / (2*smax) + (a0**2 - aB_arr[case_f]**2) / (2*jmax)
    vD_arr[case_g] = v0 - aB_arr[case_g]**2 / (2 * jmax) + (a0 + aB_arr[case_g]) * np.sqrt((a0-aB_arr[case_g]) / smax) + aB_arr[case_g] * jmax / (2*smax)
    vD_arr[case_h] = v0 + aB_arr[case_h] * np.sqrt(-aB_arr[case_h]/smax) + (a0 + aB_arr[case_h])*np.sqrt((a0 - aB_arr[case_h]) / smax)

    vD_arr[case_i] = v0 + 2*a0*abs(aB_arr[case_i])**(1/2)*(1/smax)**(1/2) - abs(aB_arr[case_i])**(3/2)*(1/smax)**(1/2)
    vD_arr[case_j] = v0 - 2*a0*abs(aB_arr[case_j])**(1/2)*(1/smax)**(1/2) + abs(aB_arr[case_j])**(3/2)*(1/smax)**(1/2)

    return aB_arr, vD_arr

def find_shortest_time(x0, xF, v0, vF, a0, aF, t0):

    aB_arr = np.linspace(-amax, amax, 5000)
    aB_arr, vD_arr_B = find_aB_vD(v0, a0, aB_arr)
    aG_arr, vD_arr_G = find_aB_vD(vF, -aF, aB_arr)

    v_min = max(np.min(vD_arr_B), np.min(vD_arr_G), -vmax)
    v_max = min(np.max(vD_arr_B), np.max(vD_arr_G), vmax)
  
    vD_interest = np.linspace(v_min, v_max, 5000)
    aB_interest = np.interp(vD_interest, vD_arr_B, aB_arr)
    aG_interest = np.interp(vD_interest, vD_arr_G, aG_arr)

    vD_B = np.interp(aB_interest, aB_arr, vD_arr_B)
    vD_G = np.interp(aG_interest, aG_arr, vD_arr_G)

    tA1, tA2, tC1, tC2, tE1, tE2, tH1, tH2, tB, tG = find_times(aB_interest, vD_B, a0, aG_interest, vD_G, -aF, vD_interest, vD_interest)
    valid_idx = tA1 >= 0
    tA1, tA2, tC1, tC2, tE1, tE2, tH1, tH2, tB, tG = tA1[valid_idx], tA2[valid_idx], tC1[valid_idx], tC2[valid_idx], tE1[valid_idx], tE2[valid_idx], tH1[valid_idx], tH2[valid_idx], tB[valid_idx], tG[valid_idx]
    
    vD_interest = vD_interest[valid_idx]
    vD_B = vD_B[valid_idx]
    vD_G = vD_G[valid_idx]
    aB_interest = aB_interest[valid_idx]
    aG_interest = aG_interest[valid_idx]

    xC = find_xC(tA1, tA2, tC1, tC2, tB, x0, v0, a0, aB_interest)
    xE = 2*xF - find_xC(tH1, tH2, tE1, tE2, tG, xF, vF, -aF, aG_interest)

    delta_X = xE - xC
    
    if delta_X[0] * delta_X[-1] < 0 and delta_X[0] > delta_X[-1]:
        vD_opt = np.interp(0, np.flip(delta_X), np.flip(vD_interest))
        tD = 0
    elif delta_X[0] * delta_X[-1] < 0:
        vD_opt = np.interp(0, delta_X, vD_interest)
        tD = 0
    elif delta_X[0] > 0:
        vD_opt = vD_interest[-1]
        tD = abs(delta_X[-1]) / abs(vD_opt)
    else:
        vD_opt = vD_interest[0]
        tD = abs(delta_X[0]) / abs(vD_opt)

    aB = np.interp(vD_opt, vD_arr_B, aB_arr)
    aG = np.interp(vD_opt, vD_arr_G, aG_arr)

    if abs(aB) < abs(a0):
        aB = a0
    
    if abs(aG) < abs(aF):
        aG = -aF

    _, vD_tB0_b = find_aB_vD(v0, a0, np.array([aB]))
    _, vD_tB0_g = find_aB_vD(vF, -aF, np.array([aG]))

    tA1_opt, tA2_opt, tC1_opt, tC2_opt, tE1_opt, tE2_opt, tH1_opt, tH2_opt, tB_opt, tG_opt = find_times(np.array([aB]), np.array([vD_opt]), a0, np.array([aG]), np.array([vD_opt]), -aF, vD_tB0_b, vD_tB0_g)
    times_list = [2*tA1_opt[0], tA2_opt[0], tB_opt[0], 2*tC1_opt[0], tC2_opt[0], tD, 2*tE1_opt[0], tE2_opt[0], tG_opt[0], 2*tH1_opt[0], tH2_opt[0]]

    time_stamps = { 'tA1': tA1_opt[0],
                    'tA2': tA2_opt[0],
                    'tC1': tC1_opt[0],
                    'tC2': tC2_opt[0],
                    'tB' : tB_opt[0],
                    'tD' : tD,
                    'tE1': tE1_opt[0],
                    'tE2': tE2_opt[0],
                    'tH1': tH1_opt[0],
                    'tH2': tH2_opt[0],
                    'tG' : tG_opt[0]}

    valid_idx = np.sign(delta_X) == np.sign(vD_interest)
    
    delta_X = delta_X[valid_idx]
    vD_interest = vD_interest[valid_idx]

    tD_arr = abs(delta_X / vD_interest)
    
    T_arr = 2*tA1[valid_idx] + tA2[valid_idx] + tB[valid_idx] + 2*tC1[valid_idx] + tC2[valid_idx] + tD_arr + 2*tE1[valid_idx] + tE2[valid_idx] + tG[valid_idx] + 2*tH1[valid_idx] + tH2[valid_idx] 

    return sum(times_list), T_arr, vD_interest, vD_B[valid_idx], vD_G[valid_idx], aB_interest[valid_idx], aG_interest[valid_idx], Trajectory([x0, v0, a0], [xF, vF, aF], time_stamps, aB, aG, t0)


def find_times(aB, vD_b, a0, aG, vD_g, aF, vD_tB0_b, vD_tB0_g):
    tA1, tA2 = find_times_single(np.abs(aB-a0))
    tC1, tC2 = find_times_single(np.abs(aB))
    tE1, tE2 = find_times_single(np.abs(aG))
    tH1, tH2 = find_times_single(np.abs(aG-aF))

    aB_maxed = (aB == amax) | (aB == -amax) 
    aB_a0 = aB == a0

    tB = np.zeros_like(aB)
    tB[(~aB_maxed) & (~aB_a0)] = 0
    tB[aB_maxed] = np.abs((vD_tB0_b[aB_maxed] - vD_b[aB_maxed]) / amax)
    tB[aB_a0] = np.abs((vD_tB0_b[aB_a0] - vD_b[aB_a0]) / a0)

    aG_maxed = (aG == amax) | (aG == -amax)
    aG_aF = aG == aF
 
    tG = np.zeros_like(aG)
    tG[~aG_maxed & ~aG_aF] = 0
    tG[aG_maxed] = np.abs((vD_tB0_g[aG_maxed] - vD_g[aG_maxed]) / amax)
    tG[aG_aF] = np.abs((vD_tB0_g[aG_aF] - vD_g[aG_aF]) / aF)

    return tA1, tA2, tC1, tC2, tE1, tE2, tH1, tH2, tB, tG

def find_times_single(topic):
    t1 = np.zeros_like(topic)
    t2 = np.zeros_like(topic)
    case_t = topic > crit_value

    t1[case_t] = jmax / smax
    t2[case_t] = topic[case_t] / jmax - jmax / smax
    t1[~case_t] = np.sqrt(topic[~case_t] / smax)
    t2[~case_t] = 0

    return t1, t2

def create_trajectory_from_vertices(vertices, optimize=True):
    temp_path = [[], [], []]
    smooth_path = [[], [], []]
    t_complete = [0]
    t_temp = [0]
    t0 = 0

    for vertex in vertices[1:]:
        start = np.r_['0,2', vertex.parent_vertex.state[0:3], [0,0,0], [0,0,0]]
        end = np.r_['0,2', vertex.state[0:3], [0,0,0], [0,0,0]]

        print(vertex.state[3])
        x_traj, y_traj, z_traj = find_kinodynamic_trajectory(start, end, t0, vertex.state[3]-t0)
        t_end_traj = x_traj.t0 + x_traj.target_time
        t0 = t_end_traj
        t_temp.append(t_end_traj)
        
        temp_path[0].append(x_traj)
        temp_path[1].append(y_traj)
        temp_path[2].append(z_traj)
    
    if not optimize:
        return temp_path, t_temp

    for idx in range(len(temp_path[0][:-1])):
        curr_path_x = temp_path[0][idx]
        curr_path_y = temp_path[1][idx]
        curr_path_z = temp_path[2][idx]
        next_path_x = temp_path[0][idx+1]
        next_path_y = temp_path[1][idx+1]
        next_path_z = temp_path[2][idx+1]

        t0 = t_complete[-1]
        t_cross = curr_path_x.t0 + curr_path_x.target_time
        t_end = next_path_x.t0 + next_path_x.target_time
    
        # x1 = t_cross - 1.5
        # x2 = min(t_cross + 1.3, (t_cross + t_end) / 2)
        ratio = 0.3
        x1 = t_cross - min((t_cross-t0) * ratio, 2)
        x2 = t_cross + min((t_end-t_cross) * ratio, 2)
        # print(x1, x2)

        if len(smooth_path[0]) == 0:
            state0 = np.asarray([curr_path_x.evaluate_track(t0)[:3], curr_path_y.evaluate_track(t0)[:3], curr_path_z.evaluate_track(t0)[:3]]).T
        else:
            state0 = np.asarray([smooth_path[0][-1].evaluate_track(t0)[:3], smooth_path[1][-1].evaluate_track(t0)[:3], smooth_path[2][-1].evaluate_track(t0)[:3]]).T
            
        state1 = np.asarray([curr_path_x.evaluate_track(x1)[:3], curr_path_y.evaluate_track(x1)[:3], curr_path_z.evaluate_track(x1)[:3]]).T
        state2 = np.asarray([next_path_x.evaluate_track(x2)[:3], next_path_y.evaluate_track(x2)[:3], next_path_z.evaluate_track(x2)[:3]]).T
        state3 = np.asarray([next_path_x.evaluate_track(t_end)[:3], next_path_y.evaluate_track(t_end)[:3], next_path_z.evaluate_track(t_end)[:3]]).T

        x_traj1, y_traj1, z_traj1 = find_kinodynamic_trajectory(state0, state1, t0) 
        # plot_trajectories([x_traj1, y_traj1, z_traj1])
        t1_end = x_traj1.t0 + x_traj1.target_time
        # print(t1_end)

        x_traj2, y_traj2, z_traj2 = find_kinodynamic_trajectory(state1, state2, t1_end)
        # plot_trajectories([x_traj2, y_traj2, z_traj2]) 
        t2_end = x_traj2.t0 + x_traj2.target_time
        

        if idx + 2 == len(temp_path[0]):
            x_traj3, y_traj3, z_traj3 = find_kinodynamic_trajectory(state2, state3, t2_end) 
            # plot_trajectories([x_traj3, y_traj3, z_traj3]) 
            t3_end = x_traj3.t0 + x_traj3.target_time

            t_complete.extend([t1_end, t2_end, t3_end])
            smooth_path[0].extend([x_traj1, x_traj2, x_traj3])
            smooth_path[1].extend([y_traj1, y_traj2, y_traj3])
            smooth_path[2].extend([z_traj1, z_traj2, z_traj3])
        else:
            t_complete.extend([t1_end, t2_end])
            smooth_path[0].extend([x_traj1, x_traj2])
            smooth_path[1].extend([y_traj1, y_traj2])
            smooth_path[2].extend([z_traj1, z_traj2])

    return smooth_path, t_complete

d_arr = np.linspace(0, 17.5, 1000)
t_arr = np.zeros_like(d_arr)

for idx, d in enumerate(d_arr):
    t_arr[idx], _, _, _, _, _, _, _ = find_shortest_time(0, d, 0, 0, 0, 0, 0)

def min_time(x0, xF):
    d = np.linalg.norm(xF-x0)
    return np.interp(d, d_arr, t_arr)

if __name__ == '__main__':

    
    start = np.array([[ 8.91764331,  2.87771418,  4.09435782],
                      [-0.0226726,  -1.99999971,  0.96070989],
                      [ 0.       ,   0.        ,  0.        ]])

    end = np.array([[ 7.40833365,  0.91027013,  6.90419437],
                      [-1.55301091, -0.10880612,  1.99999971],
                      [ 0.        ,  0.        ,  0.        ]])

    t0 = 0
    # plot_trajectories(find_kinodynamic_trajectory(start, end, t0))
    plt.figure()
    plt.plot(d_arr, t_arr)
    plt.show()