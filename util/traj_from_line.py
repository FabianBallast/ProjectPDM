import numpy as np

def point_from_traj(start_pos, end_pos, time_ttl, t_c, t0):
    v_max = (end_pos-start_pos)*2/(time_ttl - t0)
    if t_c >= t0 and t_c - t0 < (time_ttl-t0)/2:
        vel = 2*v_max*(t_c-t0)/(time_ttl - t0)
        pos = start_pos + (t_c-t0)*vel/2
        acc = 0*start_pos
    elif t_c < t0:
        pos = start_pos
        vel = 0*start_pos
        acc = 0*start_pos
    elif t_c > time_ttl:
        pos= end_pos
        vel = 0*start_pos
        acc = 0*start_pos
    else:
        vel = 2*v_max*(time_ttl-t_c)/(time_ttl-t0)
        pos = end_pos - (time_ttl-t_c)*vel/2
        acc = 0*start_pos

    return (pos, vel, acc)

def fall_down(start_pos, end_pos, maxAcc, t_c, t0):
    dist = np.linalg.norm(end_pos-start_pos)
    t1 = np.sqrt(dist * maxAcc / 9.81**2) + t0
    t2 = 9.81/maxAcc * (t1-t0) + t1

    if t_c >= t0 and t_c <= t1:
        acc = np.array([0,0,-9.81])
        vel = acc*(t_c-t0)
        pos = start_pos + (t_c-t0)*vel/2
    elif t_c < t0:
        pos = start_pos
        vel = 0*start_pos
        acc = 0*start_pos
    elif t_c > t2:
        pos= end_pos
        vel = 0*start_pos
        acc = 0*start_pos
    else:
        acc = np.array([0,0,maxAcc])
        vel = acc*(t_c-t2)
        pos = end_pos + (t_c-t2)*vel/2
    return (pos, vel, acc, t2)