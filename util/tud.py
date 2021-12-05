import numpy as np
import util.traj_from_line as traj

def tud(t):

    point11 = np.array([0,0,0])
    point10 = np.array([0,0,7])
    point9 = np.array([0,-2,7])
    point8 = np.array([0,3,7])
    point7 = np.array([0,3,2])
    point6 = np.array([0,7,2])
    point5 = np.array([0,7,7])
    point4 = np.array([0,8.5,7])
    point3 = np.array([0,8.5,0])
    point2 = np.array([0,8,0])
    point1 = np.array([0,8,7])

    # _, _, _, T1 = traj.fall_down(point1, point2, 9.81, 0, 0)

    T1 = 1.8 
    T2 = T1 + 0.3
    T3 = np.pi * 3.5 * (T2-T1) + T2
    T4 = 3*(T2-T1) + T3 
    T5 = T4+2.3
    T6 = T5+np.pi*2/10*(T5-T4)
    T7 = T6+T5-T4
    T8 = T7+1.8 
    T9 = T8+1.7 
    T10 = T9+1.9

    # print(T5, T6, T7, T10)

    dt = 0.0001 
    done = False

    if t >= T10:
        pos = point11 
        vel = np.array([0,0,0])
        acc = np.array([0,0,0])
        done = True
    elif t < T1:
        # pos, vel, acc, _ = traj.fall_down(point1, point2, 9.81, t, 0)#traj.point_from_traj(point1, point2, T1, t, 0) 
        pos, vel, acc = traj.point_from_traj(point1, point2, T1, t, 0) 
        pos_dt, _, _ = traj.point_from_traj(point1, point2, T1, t+dt, 0) 
        pos_2dt, _, _ = traj.point_from_traj(point1, point2, T1, t+2*dt, 0) 
        vel = (pos_dt - pos)/dt 
        acc = (pos_2dt - 2*pos_dt + pos) / dt**2 
    elif t < T2:
        pos, _, _ = traj.point_from_traj(point2, 2*point3-point2, 2*T2-T1, t, T1) 
        pos_dt, _, _ = traj.point_from_traj(point2, 2*point3-point2, 2*T2-T1, t+dt, T1) 
        pos_2dt, _, _ = traj.point_from_traj(point2, 2*point3-point2, 2*T2-T1, t+2*dt, T1) 
        vel = (pos_dt - pos)/dt 
        acc = (pos_2dt - 2*pos_dt + pos) / dt**2 
    elif t < T3:
        radius = 3.5
        v = np.pi * radius / (T3-T2)
        angle = -np.pi/2 + (t-T2)/(T3-T2) * np.pi
        pos = point3 + radius*np.array([0, np.cos(angle),  np.sin(angle)+1] )
        vel = v*np.array([0,-np.sin(angle),  np.cos(angle)]) 
        acc = v**2/radius*np.array([0,-np.cos(angle),  -np.sin(angle)] )
    elif t < T4:
        pos, _, _ = traj.point_from_traj(2*point4-point5, point5, T4, t, 2*T3-T4) 
        pos_dt, _, _ = traj.point_from_traj(2*point4-point5, point5, T4, t+dt, 2*T3-T4) 
        pos_2dt, _, _ = traj.point_from_traj(2*point4-point5, point5, T4, t+2*dt, 2*T3-T4) 
        vel = (pos_dt - pos)/dt 
        acc = (pos_2dt - 2*pos_dt + pos) / dt**2 
    elif t < T5:
        pos, _, _ = traj.point_from_traj(point5, 2*point6-point5, 2*T5-T4, t, T4) 
        pos_dt, _, _ = traj.point_from_traj(point5, 2*point6-point5, 2*T5-T4, t+dt, T4) 
        pos_2dt, _, _ = traj.point_from_traj(point5, 2*point6-point5, 2*T5-T4, t+2*dt, T4) 
        vel = (pos_dt - pos)/dt 
        acc = (pos_2dt - 2*pos_dt + pos) / dt**2 
    elif t < T6:
        radius = 2
        v = np.pi * radius / (T6-T5)
        angle = -(t-T5)/(T6-T5) * np.pi
        pos = point6 + radius*np.array([0, np.cos(angle)-1,  np.sin(angle)] )
        vel = -v*np.array([0,-np.sin(angle),  np.cos(angle)]) 
        acc = v**2/radius*np.array([0,-np.cos(angle),  -np.sin(angle)] )
    elif t < T7:
        pos, _, _ = traj.point_from_traj(2*point7-point8, point8, T7, t, 2*T6-T7) 
        pos_dt, _, _ = traj.point_from_traj(2*point7-point8, point8, T7, t+dt, 2*T6-T7) 
        pos_2dt, _, _ = traj.point_from_traj(2*point7-point8, point8, T7, t+2*dt, 2*T6-T7) 
        vel = (pos_dt - pos)/dt 
        acc = (pos_2dt - 2*pos_dt + pos) / dt**2 
    elif t < T8:
        pos, _, _ = traj.point_from_traj(point8, point9, T8, t, T7) 
        pos_dt, _, _ = traj.point_from_traj(point8, point9, T8, t+dt, T7) 
        pos_2dt, _, _ = traj.point_from_traj(point8, point9, T8, t+2*dt, T7) 
        vel = (pos_dt - pos)/dt 
        acc = (pos_2dt - 2*pos_dt + pos) / dt**2 
    elif t < T9:
        pos, _, _ = traj.point_from_traj(point9, point10, T9, t, T8) 
        pos_dt, _, _ = traj.point_from_traj(point9, point10, T9, t+dt, T8) 
        pos_2dt, _, _ = traj.point_from_traj(point9, point10, T9, t+2*dt, T8) 
        vel = (pos_dt - pos)/dt 
        acc = (pos_2dt - 2*pos_dt + pos) / dt**2 
    elif t < T10:
        pos, _, _ = traj.point_from_traj(point10, point11, T10, t, T9) 
        pos_dt, _, _ = traj.point_from_traj(point10, point11, T10, t+dt, T9) 
        pos_2dt, _, _ = traj.point_from_traj(point10, point11, T10, t+2*dt, T9) 
        vel = (pos_dt - pos)/dt 
        acc = (pos_2dt - 2*pos_dt + pos) / dt**2 
    

    yaw = 0 
    yawdot = 0 

    return pos, vel, acc, yaw, yawdot, done
