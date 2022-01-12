import numpy as np
import matplotlib.pyplot as plt

vmax = 1.5
amax = 4
jmax = 12
smax = 50

crit_value = jmax**2/smax

class Trajectory:
    def __init__(self, initial_state, target_state, time_stamps, aB, aG, t0=0, reverse=False):

        if not reverse:
            self.initial_state = initial_state
            self.target_state  = target_state
            self.time_stamps = time_stamps
            self.cum_time = np.cumsum([self.time_stamps['tA1'], self.time_stamps['tA2'], self.time_stamps['tA1'], self.time_stamps['tB'],
                                    self.time_stamps['tC1'], self.time_stamps['tC2'], self.time_stamps['tC1'], self.time_stamps['tD'],
                                    self.time_stamps['tE1'], self.time_stamps['tE2'], self.time_stamps['tE1'], self.time_stamps['tG'],
                                    self.time_stamps['tH1'], self.time_stamps['tH2'], self.time_stamps['tH1']])
            
            self.reverse = Trajectory(initial_state, target_state, time_stamps, aB, aG, t0=t0, reverse=True)
            self.s1, self.s4 = np.sign(aB-initial_state[2])*smax, -np.sign(aB)*smax
            self.aB, self.aG = aB, aG
            self.direction = 1
        else:
            self.initial_state = [target_state[0], target_state[1], -target_state[2]]
            self.target_state  = initial_state
            self.time_stamps = {'tA1': time_stamps['tH1'],
                                'tA2': time_stamps['tH2'],
                                'tC1': time_stamps['tE1'],
                                'tC2': time_stamps['tE2'],
                                'tB' : time_stamps['tG'],
                                'tD' : time_stamps['tD'],
                                'tE1': time_stamps['tC1'],
                                'tE2': time_stamps['tC2'],
                                'tH1': time_stamps['tA1'],
                                'tH2': time_stamps['tA2'],
                                'tG' : time_stamps['tB']}
            self.cum_time = np.cumsum([self.time_stamps['tA1'], self.time_stamps['tA2'], self.time_stamps['tA1'], self.time_stamps['tB'],
                                    self.time_stamps['tC1'], self.time_stamps['tC2'], self.time_stamps['tC1'], self.time_stamps['tD'],
                                    self.time_stamps['tE1'], self.time_stamps['tE2'], self.time_stamps['tE1'], self.time_stamps['tG'],
                                    self.time_stamps['tH1'], self.time_stamps['tH2'], self.time_stamps['tH1']])
            
            self.s1, self.s4 = np.sign(aG-target_state[2])*smax, -np.sign(aG)*smax
            self.aB, self.aG = aG, aB
            self.direction = -1
        
        self.target_time = self.cum_time[-1]
        self.t0 = t0

        if self.cum_time[0] > 0:
            self.t1 = self.cum_time[2]
        elif self.cum_time[3] > 0:
            self.t1 = self.cum_time[3]
        else:
            self.t1 = self.cum_time[6]
        
        if not reverse:
            if self.reverse.cum_time[0] > 0:
                self.t2 = self.cum_time[2]
            elif self.reverse.cum_time[3] > 0:
                self.t2 = self.cum_time[3]
            else:
                self.t2 = self.cum_time[6]

    def evaluate_track(self, t):
        t_net = t - self.t0

        # Part A
        if t_net < self.cum_time[0]:
            return self.__track_A1(t_net)
        elif t_net < self.cum_time[1]:
            return self.__track_A2(t_net)
        elif t_net < self.cum_time[2]:
            return self.__track_A3(t_net)

        # Part B
        elif t_net < self.cum_time[3]:
            return self.__track_B(t_net)
        
        # Part C
        elif t_net < self.cum_time[4]:
            return self.__track_C1(t_net)
        elif t_net < self.cum_time[5]:
            return self.__track_C2(t_net)
        elif t_net < self.cum_time[6]:
            return self.__track_C3(t_net)

        # Part D
        elif t_net < self.cum_time[7]:
            return self.__track_D(t_net)
        
        # Part E
        elif t_net < self.cum_time[8]:
            return self.reverse.__track_C3(self.target_time-t_net)
        elif t_net < self.cum_time[9]:
            return self.reverse.__track_C2(self.target_time-t_net)
        elif t_net < self.cum_time[10]:
            return self.reverse.__track_C1(self.target_time-t_net)
        
        # Part G
        elif t_net < self.cum_time[11]:
            return self.reverse.__track_B(self.target_time-t_net)
        
        # Part H
        elif t_net < self.cum_time[12]:
            return self.reverse.__track_A3(self.target_time-t_net)
        elif t_net < self.cum_time[13]:
            return self.reverse.__track_A2(self.target_time-t_net)
        elif t_net < self.cum_time[14]:
            return self.reverse.__track_A1(self.target_time-t_net)
        
        else:
            return self.reverse.__track_A1(0)


    def __track_A1(self, t):
        x = ((self.s1*t**4)/24 + (self.initial_state[2]*t**2)/2 + self.initial_state[1]*t) * self.direction + self.initial_state[0]
        v = (self.s1*t**3)/6 + self.initial_state[2]*t + self.initial_state[1]
        a = (self.initial_state[2] + 0.5 * self.s1 * t**2) * self.direction
        j = self.s1 * t
        return np.array([x, v, a, j])
    
    def __track_A2(self, t):
        x_p, v_p, a_p, j_p = self.__track_A1(self.cum_time[0])
        x = x_p + self.direction * ((t - self.time_stamps['tA1'])*(2*self.s1*t**2*self.time_stamps['tA1'] - self.s1*t*self.time_stamps['tA1']**2 + 6*self.initial_state[2]*t + self.s1*self.time_stamps['tA1']**3 + 6*self.initial_state[2]*self.time_stamps['tA1'] + 12*self.initial_state[1]))/12
        v = v_p + ((t - self.time_stamps['tA1'])*(2*self.initial_state[2] + self.s1*t*self.time_stamps['tA1']))/2
        a = a_p + self.direction * self.s1*self.time_stamps['tA1']*(t - self.time_stamps['tA1'])
        j = j_p
        return np.array([x, v, a, j])

    def __track_A3(self, t):
        x_p, v_p, a_p, j_p = self.__track_A2(self.cum_time[1])
        x = x_p - self.direction * ((self.time_stamps['tA1'] - t + self.time_stamps['tA2'])*(- self.s1*t**3 + 7*self.s1*t**2*self.time_stamps['tA1'] + 3*self.s1*t**2*self.time_stamps['tA2'] - 5*self.s1*t*self.time_stamps['tA1']**2 - 2*self.s1*t*self.time_stamps['tA1']*self.time_stamps['tA2'] - \
            3*self.s1*t*self.time_stamps['tA2']**2 + 12*self.initial_state[2]*t + 3*self.s1*self.time_stamps['tA1']**3 + 5*self.s1*self.time_stamps['tA1']**2*self.time_stamps['tA2'] + 7*self.s1*self.time_stamps['tA1']*self.time_stamps['tA2']**2 + 12*self.initial_state[2]*self.time_stamps['tA1'] + self.s1*self.time_stamps['tA2']**3 + 12*self.initial_state[2]*self.time_stamps['tA2'] + 24*self.initial_state[1]))/24
        v = v_p - ((self.time_stamps['tA1'] - t + self.time_stamps['tA2'])*(- self.s1*t**2 + 5*self.s1*t*self.time_stamps['tA1'] + 2*self.s1*t*self.time_stamps['tA2'] - self.s1*self.time_stamps['tA1']**2 + self.s1*self.time_stamps['tA1']*self.time_stamps['tA2'] - self.s1*self.time_stamps['tA2']**2 + 6*self.initial_state[2]))/6
        a = a_p - self.direction * (self.s1*(self.time_stamps['tA1'] - t + self.time_stamps['tA2'])*(3*self.time_stamps['tA1'] - t + self.time_stamps['tA2']))/2
        j = j_p + self.s1*(self.time_stamps['tA1'] - t + self.time_stamps['tA2'])
        return np.array([x, v, a, j])
    
    def __track_B(self, t):
        x_p, v_p, a_p, j_p = self.__track_A3(self.cum_time[2])
        x = x_p - self.direction * ((2*self.time_stamps['tA1'] - t + self.time_stamps['tA2'])*(2*self.initial_state[1] + 4*self.initial_state[2]*self.time_stamps['tA1'] + 2*self.initial_state[2]*self.time_stamps['tA2'] + self.aB*t - 2*self.aB*self.time_stamps['tA1'] - self.aB*self.time_stamps['tA2'] + 2*self.s1*self.time_stamps['tA1']**3 + self.s1*self.time_stamps['tA1']*self.time_stamps['tA2']**2 + 3*self.s1*self.time_stamps['tA1']**2*self.time_stamps['tA2']))/2
        v = v_p - self.aB*(2*self.time_stamps['tA1'] - t + self.time_stamps['tA2'])
        a = self.direction * self.aB
        j = 0
        return np.array([x, v, a, j])
    
    def __track_C1(self, t):
        x_p, v_p, a_p, j_p = self.__track_B(self.cum_time[3])
        x = x_p - self.direction * ((2*self.time_stamps['tA1'] - t + self.time_stamps['tA2'] + self.time_stamps['tB'])*(24*self.initial_state[1] + 48*self.initial_state[2]*self.time_stamps['tA1'] + 24*self.initial_state[2]*self.time_stamps['tA2'] + 12*self.aB*t - 24*self.aB*self.time_stamps['tA1'] - 12*self.aB*self.time_stamps['tA2'] + 12*self.aB*self.time_stamps['tB'] + self.s4*t**3 + 24*self.s1*self.time_stamps['tA1']**3 - \
            8*self.s4*self.time_stamps['tA1']**3 - self.s4*self.time_stamps['tA2']**3 - self.s4*self.time_stamps['tB']**3 + 12*self.s4*t*self.time_stamps['tA1']**2 - 6*self.s4*t**2*self.time_stamps['tA1'] + 3*self.s4*t*self.time_stamps['tA2']**2 - 3*self.s4*t**2*self.time_stamps['tA2'] + 3*self.s4*t*self.time_stamps['tB']**2 - 3*self.s4*t**2*self.time_stamps['tB'] + 12*self.s1*self.time_stamps['tA1']*self.time_stamps['tA2']**2 + \
            36*self.s1*self.time_stamps['tA1']**2*self.time_stamps['tA2'] - 6*self.s4*self.time_stamps['tA1']*self.time_stamps['tA2']**2 - 12*self.s4*self.time_stamps['tA1']**2*self.time_stamps['tA2'] - 6*self.s4*self.time_stamps['tA1']*self.time_stamps['tB']**2 - 12*self.s4*self.time_stamps['tA1']**2*self.time_stamps['tB'] - 3*self.s4*self.time_stamps['tA2']*self.time_stamps['tB']**2 - 3*self.s4*self.time_stamps['tA2']**2*self.time_stamps['tB'] + 12*self.s4*t*self.time_stamps['tA1']*self.time_stamps['tA2'] + 12*self.s4*t*self.time_stamps['tA1']*self.time_stamps['tB'] + 6*self.s4*t*self.time_stamps['tA2']*self.time_stamps['tB'] - 12*self.s4*self.time_stamps['tA1']*self.time_stamps['tA2']*self.time_stamps['tB']))/24
        v = v_p + -((2*self.time_stamps['tA1'] - t + self.time_stamps['tA2'] + self.time_stamps['tB'])*(self.s4*t**2 - 4*self.s4*t*self.time_stamps['tA1'] - 2*self.s4*t*self.time_stamps['tA2'] - 2*self.s4*t*self.time_stamps['tB'] + 4*self.s4*self.time_stamps['tA1']**2 + 4*self.s4*self.time_stamps['tA1']*self.time_stamps['tA2'] + 4*self.s4*self.time_stamps['tA1']*self.time_stamps['tB'] + self.s4*self.time_stamps['tA2']**2 + 2*self.s4*self.time_stamps['tA2']*self.time_stamps['tB'] + self.s4*self.time_stamps['tB']**2 + 6*self.aB))/6
        a = a_p + self.direction * (self.s4*(2*self.time_stamps['tA1'] - t + self.time_stamps['tA2'] + self.time_stamps['tB'])**2)/2
        j = j_p + self.s4 * (t-(2*self.time_stamps['tA1']+self.time_stamps['tA2']+self.time_stamps['tB']))
        return np.array([x, v, a, j])
    
    def __track_C2(self, t):
        x_p, v_p, a_p, j_p = self.__track_C1(self.cum_time[4])
        x = x_p - self.direction * ((2*self.time_stamps['tA1'] - t + self.time_stamps['tA2'] + self.time_stamps['tB'] + self.time_stamps['tC1'])*(12*self.initial_state[1] + 24*self.initial_state[2]*self.time_stamps['tA1'] + 12*self.initial_state[2]*self.time_stamps['tA2'] + 6*self.aB*t - 12*self.aB*self.time_stamps['tA1'] - 6*self.aB*self.time_stamps['tA2'] + 6*self.aB*self.time_stamps['tB'] + 6*self.aB*self.time_stamps['tC1'] + \
               12*self.s1*self.time_stamps['tA1']**3 + self.s4*self.time_stamps['tC1']**3 - self.s4*t*self.time_stamps['tC1']**2 + 2*self.s4*t**2*self.time_stamps['tC1'] + 6*self.s1*self.time_stamps['tA1']*self.time_stamps['tA2']**2 + 18*self.s1*self.time_stamps['tA1']**2*self.time_stamps['tA2'] + 2*self.s4*self.time_stamps['tA1']*self.time_stamps['tC1']**2 + 8*self.s4*self.time_stamps['tA1']**2*self.time_stamps['tC1'] + self.s4*self.time_stamps['tA2']*self.time_stamps['tC1']**2 + \
               2*self.s4*self.time_stamps['tA2']**2*self.time_stamps['tC1'] + self.s4*self.time_stamps['tB']*self.time_stamps['tC1']**2 + 2*self.s4*self.time_stamps['tB']**2*self.time_stamps['tC1'] - 8*self.s4*t*self.time_stamps['tA1']*self.time_stamps['tC1'] - 4*self.s4*t*self.time_stamps['tA2']*self.time_stamps['tC1'] - 4*self.s4*t*self.time_stamps['tB']*self.time_stamps['tC1'] + 8*self.s4*self.time_stamps['tA1']*self.time_stamps['tA2']*self.time_stamps['tC1'] + 8*self.s4*self.time_stamps['tA1']*self.time_stamps['tB']*self.time_stamps['tC1'] + 4*self.s4*self.time_stamps['tA2']*self.time_stamps['tB']*self.time_stamps['tC1']))/12
        v = v_p + ((2*self.s4*self.time_stamps['tA1']*self.time_stamps['tC1'] - self.s4*t*self.time_stamps['tC1'] - 2*self.aB + self.s4*self.time_stamps['tA2']*self.time_stamps['tC1'] + self.s4*self.time_stamps['tB']*self.time_stamps['tC1'])*(2*self.time_stamps['tA1'] - t + self.time_stamps['tA2'] + self.time_stamps['tB'] + self.time_stamps['tC1']))/2
        a = a_p - self.direction * (self.s4*self.time_stamps['tC1']*(2*self.time_stamps['tA1'] - t + self.time_stamps['tA2'] + self.time_stamps['tB'] + self.time_stamps['tC1']))
        j = j_p
        return np.array([x, v, a, j])

    def __track_C3(self, t):
        x_p, v_p, a_p, j_p = self.__track_C2(self.cum_time[5])
        x = x_p - self.direction * ((2*self.time_stamps['tA1'] - t + self.time_stamps['tA2'] + self.time_stamps['tB'] + self.time_stamps['tC1'] + self.time_stamps['tC2'])*(24*self.initial_state[1] + 48*self.initial_state[2]*self.time_stamps['tA1'] + 24*self.initial_state[2]*self.time_stamps['tA2'] + 12*self.aB*t - 24*self.aB*self.time_stamps['tA1'] - 12*self.aB*self.time_stamps['tA2'] + 12*self.aB*self.time_stamps['tB'] + \
               12*self.aB*self.time_stamps['tC1'] + 12*self.aB*self.time_stamps['tC2'] - self.s4*t**3 + 24*self.s1*self.time_stamps['tA1']**3 + 8*self.s4*self.time_stamps['tA1']**3 + self.s4*self.time_stamps['tA2']**3 + self.s4*self.time_stamps['tB']**3 + 3*self.s4*self.time_stamps['tC1']**3 + self.s4*self.time_stamps['tC2']**3 - 12*self.s4*t*self.time_stamps['tA1']**2 + 6*self.s4*t**2*self.time_stamps['tA1'] - \
               3*self.s4*t*self.time_stamps['tA2']**2 + 3*self.s4*t**2*self.time_stamps['tA2'] - 3*self.s4*t*self.time_stamps['tB']**2 + 3*self.s4*t**2*self.time_stamps['tB'] - 5*self.s4*t*self.time_stamps['tC1']**2 + 7*self.s4*t**2*self.time_stamps['tC1'] - 3*self.s4*t*self.time_stamps['tC2']**2 + 3*self.s4*t**2*self.time_stamps['tC2'] + 12*self.s1*self.time_stamps['tA1']*self.time_stamps['tA2']**2 + \
               36*self.s1*self.time_stamps['tA1']**2*self.time_stamps['tA2'] + 6*self.s4*self.time_stamps['tA1']*self.time_stamps['tA2']**2 + 12*self.s4*self.time_stamps['tA1']**2*self.time_stamps['tA2'] + 6*self.s4*self.time_stamps['tA1']*self.time_stamps['tB']**2 + 12*self.s4*self.time_stamps['tA1']**2*self.time_stamps['tB'] + 3*self.s4*self.time_stamps['tA2']*self.time_stamps['tB']**2 + 3*self.s4*self.time_stamps['tA2']**2*self.time_stamps['tB'] + 10*self.s4*self.time_stamps['tA1']*self.time_stamps['tC1']**2 + \
               28*self.s4*self.time_stamps['tA1']**2*self.time_stamps['tC1'] + 6*self.s4*self.time_stamps['tA1']*self.time_stamps['tC2']**2 + 5*self.s4*self.time_stamps['tA2']*self.time_stamps['tC1']**2 + 12*self.s4*self.time_stamps['tA1']**2*self.time_stamps['tC2'] + 7*self.s4*self.time_stamps['tA2']**2*self.time_stamps['tC1'] + 3*self.s4*self.time_stamps['tA2']*self.time_stamps['tC2']**2 + 3*self.s4*self.time_stamps['tA2']**2*self.time_stamps['tC2'] + 5*self.s4*self.time_stamps['tB']*self.time_stamps['tC1']**2 + \
               7*self.s4*self.time_stamps['tB']**2*self.time_stamps['tC1'] + 3*self.s4*self.time_stamps['tB']*self.time_stamps['tC2']**2 + 3*self.s4*self.time_stamps['tB']**2*self.time_stamps['tC2'] + 7*self.s4*self.time_stamps['tC1']*self.time_stamps['tC2']**2 + 5*self.s4*self.time_stamps['tC1']**2*self.time_stamps['tC2'] - 12*self.s4*t*self.time_stamps['tA1']*self.time_stamps['tA2'] - 12*self.s4*t*self.time_stamps['tA1']*self.time_stamps['tB'] - 6*self.s4*t*self.time_stamps['tA2']*self.time_stamps['tB'] - \
               28*self.s4*t*self.time_stamps['tA1']*self.time_stamps['tC1'] - 12*self.s4*t*self.time_stamps['tA1']*self.time_stamps['tC2'] - 14*self.s4*t*self.time_stamps['tA2']*self.time_stamps['tC1'] - 6*self.s4*t*self.time_stamps['tA2']*self.time_stamps['tC2'] - 14*self.s4*t*self.time_stamps['tB']*self.time_stamps['tC1'] - 6*self.s4*t*self.time_stamps['tB']*self.time_stamps['tC2'] - 2*self.s4*t*self.time_stamps['tC1']*self.time_stamps['tC2'] + 12*self.s4*self.time_stamps['tA1']*self.time_stamps['tA2']*self.time_stamps['tB'] + \
               28*self.s4*self.time_stamps['tA1']*self.time_stamps['tA2']*self.time_stamps['tC1'] + 12*self.s4*self.time_stamps['tA1']*self.time_stamps['tA2']*self.time_stamps['tC2'] + 28*self.s4*self.time_stamps['tA1']*self.time_stamps['tB']*self.time_stamps['tC1'] + 12*self.s4*self.time_stamps['tA1']*self.time_stamps['tB']*self.time_stamps['tC2'] + 14*self.s4*self.time_stamps['tA2']*self.time_stamps['tB']*self.time_stamps['tC1'] + 6*self.s4*self.time_stamps['tA2']*self.time_stamps['tB']*self.time_stamps['tC2'] + 4*self.s4*self.time_stamps['tA1']*self.time_stamps['tC1']*self.time_stamps['tC2'] + 2*self.s4*self.time_stamps['tA2']*self.time_stamps['tC1']*self.time_stamps['tC2'] + 2*self.s4*self.time_stamps['tB']*self.time_stamps['tC1']*self.time_stamps['tC2']))/24
        v = v_p + ((2*self.time_stamps['tA1'] - t + self.time_stamps['tA2'] + self.time_stamps['tB'] + self.time_stamps['tC1'] + self.time_stamps['tC2'])*(self.s4*t**2 - 4*self.s4*t*self.time_stamps['tA1'] - 2*self.s4*t*self.time_stamps['tA2'] - 2*self.s4*t*self.time_stamps['tB'] - 5*self.s4*t*self.time_stamps['tC1'] - 2*self.s4*t*self.time_stamps['tC2'] + 4*self.s4*self.time_stamps['tA1']**2 + 4*self.s4*self.time_stamps['tA1']*self.time_stamps['tA2'] + 4*self.s4*self.time_stamps['tA1']*self.time_stamps['tB'] + 10*self.s4*self.time_stamps['tA1']*self.time_stamps['tC1'] + 4*self.s4*self.time_stamps['tA1']*self.time_stamps['tC2'] + self.s4*self.time_stamps['tA2']**2 + 2*self.s4*self.time_stamps['tA2']*self.time_stamps['tB'] + 5*self.s4*self.time_stamps['tA2']*self.time_stamps['tC1'] + 2*self.s4*self.time_stamps['tA2']*self.time_stamps['tC2'] + self.s4*self.time_stamps['tB']**2 + 5*self.s4*self.time_stamps['tB']*self.time_stamps['tC1'] + 2*self.s4*self.time_stamps['tB']*self.time_stamps['tC2'] + self.s4*self.time_stamps['tC1']**2 - self.s4*self.time_stamps['tC1']*self.time_stamps['tC2'] + self.s4*self.time_stamps['tC2']**2 - 6*self.aB))/6
        a = a_p - self.direction * (self.s4*(2*self.time_stamps['tA1'] - t + self.time_stamps['tA2'] + self.time_stamps['tB'] + 3*self.time_stamps['tC1'] + self.time_stamps['tC2'])*(2*self.time_stamps['tA1'] - t + self.time_stamps['tA2'] + self.time_stamps['tB'] + self.time_stamps['tC1'] + self.time_stamps['tC2']))/2
        j = j_p - self.s4 * (t-(2*self.time_stamps['tA1']+self.time_stamps['tA2']+self.time_stamps['tB']+self.time_stamps['tC1']+self.time_stamps['tC2']))
        return np.array([x, v, a, j])
    
    def __track_D(self, t):
        x_p, v_p, a_p, j_p = self.__track_C3(self.cum_time[6])
        x = x_p + v_p * (t - self.cum_time[6])
        v = v_p 
        a = 0
        j = 0
        return np.array([x, v, a, j])

def find_xC(tA1, tA2, tC1, tC2, tB, x0, v0, a0, aB):
    s1 = np.sign(aB-a0)*smax
    s4 = -np.sign(aB)*smax
    return (7*s1*tA1**4)/12 + (7*s1*tA1**3*tA2)/6 + s1*tA1**3*tB + 2*s1*tA1**3*tC1 + s1*tA1**3*tC2 + \
           (3*s1*tA1**2*tA2**2)/4 + (3*s1*tA1**2*tA2*tB)/2 + 3*s1*tA1**2*tA2*tC1 + (3*s1*tA1**2*tA2*tC2)/2 + \
            2*a0*tA1**2 + (s1*tA1*tA2**3)/6 + (s1*tA1*tA2**2*tB)/2 + s1*tA1*tA2**2*tC1 + (s1*tA1*tA2**2*tC2)/2 + \
            2*a0*tA1*tA2 + 2*a0*tA1*tB + 4*a0*tA1*tC1 + 2*a0*tA1*tC2 + 2*v0*tA1 + (a0*tA2**2)/2 + a0*tA2*tB + \
            2*a0*tA2*tC1 + a0*tA2*tC2 + v0*tA2 + (aB*tB**2)/2 + 2*aB*tB*tC1 + aB*tB*tC2 + v0*tB + (7*s4*tC1**4)/12 + \
           (7*s4*tC1**3*tC2)/6 + (3*s4*tC1**2*tC2**2)/4 + 2*aB*tC1**2 + (s4*tC1*tC2**3)/6 + 2*aB*tC1*tC2 + 2*v0*tC1 + \
            (aB*tC2**2)/2 + v0*tC2 + x0


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    init = [2, 3, 0]
    goal = [10, 3, 0]
    times = {'tA1': 0.25856,
             'tA2': 0,
             'tC1': 0.25856,
             'tC2': 0,
             'tB': 0,
             'tD': 0,
             'tE1': 0.25856,
             'tE2': 0,
             'tH1': 0.25856,
             'tH2': 0,
             'tG': 0}
    aB = 3.34277

    traj = Trajectory(init, goal, times, aB, aB)

    t = np.linspace(0, 8*0.258, 180)
    x = np.zeros_like(t)
    v = np.zeros_like(t)
    a = np.zeros_like(t)
    j = np.zeros_like(t)

    for idx, ti in enumerate(t):
        x[idx], v[idx], a[idx], j[idx] = traj.evaluate_track(ti)
    
    # print(find_xC(0.31072, 0, 0.31072, 0, 0, 2, 2, 0, aB))

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.plot(t, x)
    ax2.plot(t, v)
    ax3.plot(t, a)
    ax4.plot(t, j)
    plt.show()

def plot_trajectories(traj_list):
    for traj in traj_list:
        t0 = traj.t0
        t_end_traj = traj.t0 + traj.target_time
        t = np.linspace(t0, t_end_traj, 200)
        x = np.zeros_like(t)
        v = np.zeros_like(t)
        a = np.zeros_like(t)
        j = np.zeros_like(t)

        for idx, ti in enumerate(t):
            x[idx], v[idx], a[idx], j[idx] = traj.evaluate_track(ti)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        ax1.plot(t, x)
        ax2.plot(t, v)
        ax3.plot(t, a)
        ax4.plot(t, j)
    # plt.show()

