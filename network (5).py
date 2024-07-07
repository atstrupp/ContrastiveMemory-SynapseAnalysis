# from distutils.command.build import build
import numpy as np
import matplotlib
import mpl_toolkits
matplotlib.rcParams.update({'font.size': 8})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams["font.sans-serif"] = "Arial"
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import sympy as smp
import scipy.linalg as LA
import scipy as sp
from scipy import integrate
from scipy.interpolate import interp1d
import scipy.misc

class ladder:
    def __init__(self, loops = 1, dt = 0.0005): #loops should be odd
        self.loops = int(loops)   # loops are effectively the number of methylation values in ESA
        self.occupancy_list=range((2*self.loops)+2)
        self.num_nodes = 2 * self.loops + 2
        self.dt = dt
        self.trange = np.arange(0,5 + (2 * self.dt),self.dt)
        self.ratematrix = np.zeros((2*self.loops)+2) #initialize a rate matrix
        self.nullspace = np.asarray(1)
        self.RATE_FACTOR = 25

    def steady_state(self, M):
        """
        Calculate the steady state occupation for a given rate matrix and also
        a time series for a while after

        Input:
         M [array]: transition rate matrix

        Returns: nullspace [array] len (2 * self.loops + 2), the stationary distribution
            y1_t[array] shape (len(t), len(y0)), where each row is occupancy at that time
        """
        nullspace = LA.null_space(M) #get nullvectors x defined as Mx = 0

        if len(nullspace) == 0:
            nullspace = sp.zeros((2*self.loops)+2,1)
        if len(nullspace) > (1):
            nullspace = nullspace[:,0] #take first nullvector
        if nullspace[0] < 0:
            nullspace = nullspace * -1 #make it positive

        nullspace = np.squeeze(np.asarray(nullspace)) #makes it an array rather than  matrix
        factor = np.sum(nullspace)
        nullspace = nullspace/factor #normalize to 1

        y1_t= sp.integrate.odeint(func=self.derivfun, y0=nullspace, t = range(5))

        return nullspace, y1_t


    def sum_occupancy(self,y_t, graph = 'yes'):
        """
        Calculate occupancy of both the bottom and top rows of the occupancy matrix
        for each time point in a range. Plot them. Effectively just sums the first half of each row

        Input:
            y_t [array] size (trange, 2 * self.loops + 2)
            graph [string]: whether or not to graph

        Returns: [toplist]: list [float] length trange, occupation of top over time
        """

        tlist = self.trange
        toplist = []
        botlist = []
        for i in range(len(tlist)):
            toplist.append(np.sum(y_t[i,:self.loops+1])) #sum the 1st half of the ith row
        for i in range(len(tlist)):
            botlist.append(np.sum(y_t[i,self.loops+1:])) #sum the last half of the ith row
        diflist = list(np.subtract(toplist,botlist))
        if graph == 'yes':
            plt.plot(tlist, toplist, label = 'top')
            plt.plot(tlist, botlist, label = 'bot')
            plt.plot(tlist, diflist, label = 'top-bot')

            plt.legend()
            plt.xlabel('Time(s)')
            plt.ylabel('Node Occupancy')
            plt.title("Difference In Occupancy of top/Bottom Rungs")
            plt.show()
        return toplist


    def stable_dissipation(self, gamma, s= 0):
        """ get steady state and then find dissipation """

        M = self.create_ESA_matrix(s,gamma)
        ss = self.steady_state(self.ratematrix)[0]
        d = 0.0
        for i, _ in enumerate(ss):
            for j in range(i):
                Jplus = float(M[i,j] * ss[j])
                Jmin = float(M[j,i] * ss[i])
                if Jplus != 0:
                    d += ((Jplus-Jmin)*np.log(Jplus/Jmin))
        return d

    def evolve_matrix(self, nullspace):
        """
        Create a transition matrix and use it to evolve a state

        Returns y_t [array[float]] size (trange, num_nodes)
        """

        y_t = sp.integrate.odeint(func=self.derivfun, y0=nullspace, t = self.trange) #use previous steady state as initial condition
        return y_t

    def deltaE(self, s, m):

        deltaE = 2 * (1-m) + np.log((1 + (s/18.2))/(1 + (s/3000))) # here s is in units of micromolar of ligand
        return deltaE

    def create_ESA_matrix(self, s = 0, gamma = 1): #RF was 1000
        """
        Remaking the transition matrix found in ESA https://www.nature.com/articles/nphys2276

        Input:
            s [float]: background signal at that instant
            gamma [float]: <1, suppresses reverse flow to create circulation

        Returns [ndarray[float]]: size num_nodes^2

        """
        # define some rates per the paper
        k_r = 1     # bottom right
        k_r_rev = k_r * np.exp(1) * gamma  # bottom left
        k_b = 1 # top left
        k_b_rev = k_b * np.exp(1) * gamma  # top right
        w_0 = 0
        w_1 = 0

        max_m = 4
        self.num_nodes = 2 * max_m + 2
        self.loops = max_m
        M = np.zeros((self.num_nodes,self.num_nodes)) # Initialize matrix

        for a in range(1,max_m + 1): #remember zero index, top levels going back except for the first node
            M[a-1,a] = k_b
        for a in range (0,max_m): #top forward except top right
            M[a+1,a] = k_b_rev
        for a in range(max_m + 2,self.num_nodes ): # bottom going back except for bottom left
            M [a-1,a] = k_r_rev
        for a in range(max_m + 1, self.num_nodes - 1): #bottom going forward except last element
            M[a+1,a] = k_r

        # Set Vertical Rates
        for m in range(0,max_m + 1):
            deltaE = self.deltaE(s,m) #rates are a function of energy difference
            # set w
            if deltaE < 0:
                w_0 = self.RATE_FACTOR * np.exp(-1 * deltaE)
                w_1 = self.RATE_FACTOR
            else:
                w_0 = self.RATE_FACTOR
                w_1 = self.RATE_FACTOR * np.exp(deltaE)

            M[m + max_m + 1,m] = w_1
            M[m, m + max_m + 1] = w_0

        # Normalize diagonal elements
        for j in range (0,(2*self.loops)+2):
            flow_loss = 0
            for i in range (0,(2*self.loops)+2):
                flow_loss += M[i,j] # Sum probability of flowing elsewhere
            M[j,j] = 0 - flow_loss # Probability of staying the same is 0-that

        self.tm = M
        self.ratematrix= M
        return M

    def step_response(self, gamma, s, base = 0):
        """
        Get the response of the top occupancy to a step function

        Input:
            gamma [float] <=1, modifies some rates
            s [float]: size of step, signal after step
            base [float] : value of initial signal

        Returns: response [interp1d function]: response of top occupancy
        to step
        offset [float]: difference between start and end of response curve
        """

        self.create_ESA_matrix(base, gamma)
        nullspace, y1_t = self.steady_state(self.ratematrix)
        self.create_ESA_matrix(base + s, gamma)
        y_t = self.evolve_matrix(nullspace)
        nully = [nullspace]
        y_tot=np.vstack((nully,y_t))
        response = interp1d(self.trange, self.sum_occupancy(y_tot,'nograph'))
        offset = self.sum_occupancy(y_tot,'nograph')[-1] - self.sum_occupancy(y_tot,'nograph')[0]

        return response, offset


    def define_kernel(self, gamma, s = 200):
        """
        Take the derivative of a function and return that list. Especially to
        define delta response given step response

        Input:
            response interp1d object: curve of occupancy response to step

        Returns: deriv[lst[float]]: derivative of that, length two less than trange
        """
        self.create_ESA_matrix(gamma = gamma)
        response, o = self.step_response(gamma, s, base = 0)
        deriv = [scipy.misc.derivative(response, t, dx=self.dt, order=3) for t in self.trange[1:-1]] #deriv cant do ends

        return deriv

    def single_kernel_full_process(self, gamma, s):
        self.ratematrix = self.create_ESA_matrix(s = 0, gamma = gamma)
        kernel = self.define_kernel(gamma, s = s)
        diss = self.stable_dissipation(gamma)
        moment = self.first_moments([kernel])[0] #its a one element list
        norm_ker = self.normalize_kernels([kernel],[moment])[0]

        return kernel, diss, moment, norm_ker

    def first_moments(self,kernels):

        mom1_list  = []
        for ker in kernels:
            mom1 = np.trapz([self.dt / 5 * i * v for i, v in enumerate(ker)], dx = self.dt / 5)
            # first moment timestep to match with ecoli kernel (scale to be 1 second)
            mom1_list.append(mom1)
        return mom1_list

    def normalize_kernels(self, kernels, moments):  ##needs to be finished!
        ''' normalize kernels to all have first moment -1

        Returns [list[numpy arrays]]: normalized kernels
        '''

        kern_list = []
        for i, kernel in enumerate(kernels):
            moment = moments[i]
            newker = -1 * np.asarray(kernel) * np.abs((moment ** -1)) #normalize by first moment
            kern_list.append(newker)

        return kern_list


    def kernel_aucs(self, kernels):
        area_list = []
        for k in kernels:
            area = np.trapz(k,self.trange[1:-1],self.dt)
            area_list.append(area)

        return area_list

    def derivfun(self, y, t):
        """
        calculates derivative to be used by sp.integrate by multiplying y by the ratematrix
        """
        dydt = np.matmul(self.ratematrix,y)
        return dydt

    def get_random_kernels(self, RATE_FACTOR = 25, s = 200): # more ideal kernels but w/ randomness

        """ Calling This Gives a list of 50 kernels with random paramters,
        along with a list of their gamma factors, and AUC measured in two different ways. """


        self.dt = 0.0005
        gamma_list = []
        kerlist = []
        trapz_AUC_list = []
        riemann_AUC_list = []
        self.RATE_FACTOR = RATE_FACTOR
        s = s
        for i in range(10):
            gamma = np.random.rand()   #random float 0 to 1
            norm_ker = self.single_kernel_full_process(gamma,s)[3]
            trapz_AUC_list.append(np.trapz(norm_ker, dx = self.dt / 5))
            riemann_AUC_list.append(sum(norm_ker) / len(norm_ker))
            gamma_list.append(gamma)
            kerlist.append(norm_ker)
        for i in range(40):
            y = np.random.rand() #random coefficient
            x = np.random.randint(0,25) # random exponent
            gamma = y * (10 ** (-1 * x))
            norm_ker = self.single_kernel_full_process(gamma,s)[3]
            trapz_AUC_list.append(np.trapz(norm_ker, dx = self.dt / 5))
            riemann_AUC_list.append(sum(norm_ker) / len(norm_ker))
            gamma_list.append(gamma)
            kerlist.append(norm_ker)
        return gamma_list, kerlist, riemann_AUC_list, trapz_AUC_list