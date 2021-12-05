# from typing_extensions import ParamSpec
import numpy as np
from scipy.spatial.transform import Rotation, rotation

class controlller:
    def __init__(self):
        # hover control gains
        self.Kp = np.diag([15, 15, 30])
        self.Kd = np.diag([12, 12, 10])
        # self.Kp = np.diag([15, 15,e
        # angular control gains
        self.Kp_t = np.diag([3000, 3000, 3000])
        self.Kd_t = np.diag([300, 300, 300])
        # self.Kp_t = np.diag([3000, 3000, 3000])
        # self.Kd_t = np.diag([300, 300, 300])
        m = 0.030  # weight (in kg) with 5 vicon markers (each is about 0.25g)
        g = 9.81  # gravitational constant
        I = np.array([[1.43e-5, 0, 0],
                      [0, 1.43e-5, 0],
                      [0, 0, 2.89e-5]])  # inertial tensor in m^2 kg
        L = 0.046  # arm length in m
        self.rotor_speed_min = 0
        self.rotor_speed_max = 2500
        self.mass = m
        self.inertia = I
        self.invI = np.linalg.inv(I)
        self.g = g
        self.arm_length = L
        self.k_thrust = 2.3e-08
        self.k_drag = 7.8e-11

    def control(self, flat_output, state):
        '''
        :param desired state: pos, vel, acc, yaw, yaw_dot
        :param current state: pos, vel, euler, omega
        :return:
        '''
        cmd_motor_speeds = np.zeros((4,))
        cmd_thrust = 0
        cmd_moment = np.zeros((3,))
        cmd_q = np.zeros((4,))

        error_pos = flat_output[0] - state['x']
        error_vel = flat_output[1] - state['v']

        # page 29
        rdd_des = flat_output[2].T + self.Kd @ error_vel.T + self.Kp @ error_pos.T
    
        # Desired roll, pitch and yaw (in rad).
        # page 30 Equation 8
        A = self.g * np.array([[np.sin(flat_output[3]), np.cos(flat_output[3])],
                               [-np.cos(flat_output[3]), np.sin(flat_output[3])]])
        b = rdd_des[0:2]
        x=np.linalg.inv(A) @ b  
                  
        phi_des = x[0]
        theta_des = x[1]
        psi_des = flat_output[3]

        # calculate u1 (thrust)
        # page 30 Equation 8
        u1 = self.mass * (rdd_des[2] + self.g)
        
        quat = state['q']
        rotation = Rotation.from_quat(quat)
        angle = np.array(rotation.as_rotvec())
        # print(f"Target angles {np.array([phi_des, theta_des, psi_des])}")
        # page 31
        error_angle = np.matmul(self.Kp_t, np.array([phi_des, theta_des, psi_des]).T- angle) +\
            np.matmul(self.Kd_t, np.array([0,0, flat_output[4]]).T - state['w'].T)

        u2 = self.inertia @ error_angle
     
        gama = self.k_drag / self.k_thrust
        Len = self.arm_length
        cof_temp = np.array(
            [1, 1, 1, 1, 0, Len, 0, -Len, -Len, 0, Len, 0, gama, -gama, gama, -gama]).reshape(4, 4)

        u = np.array([u1, u2[0], u2[1], u2[2]])
        F_i = np.matmul(np.linalg.inv(cof_temp), u)
     
        for i in range(4):
            if F_i[i] < 0:
                F_i[i] = 0
                cmd_motor_speeds[i] = self.rotor_speed_min
            cmd_motor_speeds[i] = np.sqrt(F_i[i] / self.k_thrust)
            if cmd_motor_speeds[i] > self.rotor_speed_max:
                cmd_motor_speeds[i] = self.rotor_speed_max


        cmd_thrust = u1
        cmd_moment[0] = u2[0]
        cmd_moment[1] = u2[1]
        cmd_moment[2] = u2[2]


        control_input = {'cmd_motor_speeds': cmd_motor_speeds,
                         'cmd_thrust': cmd_thrust,
                         'cmd_moment': cmd_moment}
        return control_input
