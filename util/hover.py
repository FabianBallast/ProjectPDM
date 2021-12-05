import numpy as np

def hover(t):
    if t > 0:
        pos = np.array([0, 0, 1])
        vel = np.array([0, 0, 0])
        acc = np.array([0, 0, 0])
        yaw = 0
        yawdot = 0
        done = False
    else:
        pos = np.array([0, 0, 0])
        vel = np.array([0, 0, 0])
        acc = np.array([0, 0, 0])
        yaw = 0
        yawdot = 0
        done = False

    return pos, vel, acc, yaw, yawdot, done
