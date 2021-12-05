import numpy as np
import util.traj_from_line as traj

def circle(t):

    T = 12 
    radius = 5
    dt = 0.0001
    t0 = 0

    if t > T:
        pos = np.array([radius, 0, 2.5])
        vel = np.array([0, 0, 0])
        acc =np.array([0, 0, 0])
        done = True
    else:
        angle, _, _ = traj.point_from_traj(0, 2*np.pi, T, t, t0)
        pos = pos_from_angle(angle, radius)
        vel = get_vel(t, dt, t0, T, radius)
        acc = (get_vel(t+dt,dt, t0, T, radius) - vel)/dt
        done = False
    
    yaw = 0
    yawdot = 0

    return pos, vel, acc, yaw, yawdot, done

def pos_from_angle(alpha, radius):
    return np.array([radius*np.cos(alpha), radius*np.sin(alpha), 2.5*alpha/(2*np.pi)])

def get_vel(t, dt, t0, T, radius):
    angle1, _, _ = traj.point_from_traj(0, 2*np.pi, T, t, t0)
    pos1 = pos_from_angle(angle1, radius)
    angle2, _, _ = traj.point_from_traj(0, 2*np.pi, T, t+dt, t0)
    return (pos_from_angle(angle2, radius) - pos1)/dt
        