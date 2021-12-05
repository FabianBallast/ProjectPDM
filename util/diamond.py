import numpy as np
import util.traj_from_line as traj

def diamond(t):
        
    point1 = np.array([0,0,0])
    point2 = np.sqrt(np.array([0,2,2]))
    point3 = 2*np.sqrt(np.array([0,0,2]))
    point4 = np.array([0, -np.sqrt(2), np.sqrt(2)])
    point5 = np.array([1,0,0])

    T = 12
    dt = 0.0001
    done = False

    if t > T:
        pos = point5 
        vel = np.array([0,0,0])
        acc = np.array([0,0,0]) 
        done = True
    elif t < T/4:
        pos, _, _ = traj.point_from_traj(point1, point2, T/4, t, 0) 
        pos_dt, _, _ = traj.point_from_traj(point1, point2, T/4, t+dt, 0) 
        pos_2dt, _, _ = traj.point_from_traj(point1, point2, T/4, t+2*dt, 0) 
        vel = (pos_dt - pos)/dt 
        acc = (pos_2dt - 2*pos_dt + pos) / dt**2 
    elif t < T/2:
        pos, _, _ = traj.point_from_traj(point2, point3, T/2, t, T/4) 
        pos_dt, _, _ = traj.point_from_traj(point2, point3, T/2, t+dt, T/4) 
        pos_2dt, _, _ = traj.point_from_traj(point2, point3, T/2, t+2*dt, T/4) 
        vel = (pos_dt - pos)/dt 
        acc = (pos_2dt - 2*pos_dt + pos) / dt**2 
    elif t < 3*T/4:
        pos, _, _ = traj.point_from_traj(point3, point4, 3*T/4, t, T/2) 
        pos_dt, _, _ = traj.point_from_traj(point3, point4, 3*T/4, t+dt, T/2) 
        pos_2dt, _, _ = traj.point_from_traj(point3, point4, 3*T/4, t+2*dt, T/2) 
        vel = (pos_dt - pos)/dt 
        acc = (pos_2dt - 2*pos_dt + pos) / dt**2 
    else:
        pos, _, _ = traj.point_from_traj(point4, point5, T, t, 3*T/4) 
        pos_dt, _, _ = traj.point_from_traj(point4, point5, T, t+dt, 3*T/4) 
        pos_2dt, _, _ = traj.point_from_traj(point4, point5, T, t+2*dt, 3*T/4) 
        vel = (pos_dt - pos)/dt 
        acc = (pos_2dt - 2*pos_dt + pos) / dt**2 


    yaw = 0 
    yawdot = 0 

    return pos, vel, acc, yaw, yawdot, done