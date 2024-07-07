import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import convolve1d
from scipy.integrate import trapz
from scipy import stats
import matplotlib
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams["font.sans-serif"] = "Arial"
import mpl_toolkits
matplotlib.rcParams.update({'font.size': 8})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams["font.sans-serif"] = "Arial"
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import pwlf
import network as net


class Simulation:

    def __init__(self, dt = 1e-4): # dt is the timestep for numerical integration

        #Set some params
        self.dt = dt
        self.kernel_mesh_size = int(1/self.dt)
        self.kernel = self._initialize_uneven_ecoli_kernel(0)


# SECTION: SET UP CURVES

    def _initialize_uneven_ecoli_kernel(self,I):

        """ Creates a memory kernel based on the ecoli chemotaxis response in
        https://www.pnas.org/doi/10.1073/pnas.0909673107

        Input:
            I [float]: the desired area under the curve of the kernel

        Output: kernel [list]: a list size 1 * kernel_mesh_size that is the
        numerical memory kernel
        """

        # Set the exponential decay timescale
        B = 20

        # Set other constants to normalize first moment while aquiring desired AUC
        A = 3 * I * (B ** 2) + B ** 3
        C = (B ** 2 + 2 * B * I) / (6 * I + 2 * B)

        # Create Kernel
        t = np.arange(self.kernel_mesh_size)/self.kernel_mesh_size
        kernel = A * np.exp(-B * t) * (t - (C) * (t ** 2))
        self.kernel = kernel
        return kernel

    def construct_external_signal(self,free_value, clamp_value, fast_time, slow_time):

        """ Create a numerical sawtooth wave signal to feed to the memory kernel integrator

        Input:
            free_value [float]: initial value of the sawtooth wave
            clamp_value [float]: extremum value of sawtooth wave
            fast_time [float]: time at which extremum value is reached
            slow_time [float]: total time of signal including downswing

        Output: sawtooth [list]: numerical representation of the sawtooth wave signal
        """

        # create a list of times going 0 to slow_time with appropriate mesh density
        tlist = np.arange(slow_time/(self.dt))*self.dt

        # Make skeleton x and y lists to be interpolated
        x = [0,fast_time, slow_time]
        y = [free_value, clamp_value, free_value]

        #create full wave
        interpolated_sawtooth = interp1d(x,y)
        sawtooth = interpolated_sawtooth(tlist)

        return sawtooth

    def compute_period_weight_update(self, free_value, clamp_value, fast_time, slow_time, nonlinearity_threshold, nonlinearity_style = "linear_threshold", downswing ='yes', mode = 'mirror'):

        """ Convolve the memory kernel with the signal, threshold it, then integrate over a full cycle

        Input:
            free_value, clamp_value, fast_time ,slow_time : define sawtooth
            nonlinearity_threshold [float]: nonlinearity threshold
            nonlinearity_style [string]: how to apply a nonlineariy. Default value
                deletes all values less than threshold
            downswing [string]: whether to include the second part of the sawtooth
                for diagnostic purposes

        Output: ds [float]: integreal of thresholded convolution values, represnting
            an approximation of the amplitude of the sawtooth wave
        """

        # construct sawtooth wave signal
        if downswing == 'yes':
            external_signal = self.construct_external_signal(free_value, clamp_value, fast_time, slow_time)

        if downswing == 'no':
            external_signal = self.construct_external_signal(free_value, clamp_value, fast_time, slow_time = fast_time + self.dt)[:-1]

        # convolve the sawtooth with the memory kernel
        if downswing == 'yes':
            dsdt = convolve1d(external_signal,self.kernel,mode='wrap')*self.dt #####change back to wrap
        if downswing == 'no':
            dsdt = convolve1d(external_signal,self.kernel,mode= mode)*self.dt

        # apply the threshold
        g_dsdt = self.apply_nonlinearity(dsdt, nonlinearity_threshold, nonlinearity_style)

        # integrate over the whole sawtooth
        ds = trapz(g_dsdt,dx=self.dt)

        return ds

    def apply_nonlinearity(self, dsdt, threshold = 1, style="linear_threshold"):

        """ Applies the appropriate nonlinear operation

        Input:
            dsdt [list]: values to be thresholded
            threshold [float]: threshold value
            style [str]: what type of nonlinearity to apply

        Output: dsdt [list]: input list after having been thresholded
        """

        # remove small values
        if style == "linear_threshold":
            dsdt[np.abs(dsdt) < threshold] = 0

        # cube values
        if style == "cubic":
            for i in range(len(dsdt)):
                dsdt[i] = ((dsdt[i]**3))/threshold

        return dsdt


    def get_RAND_kernels(self, RATE_FACTOR = 25, s = 200):
        '''
        get ladder network kernels, gamma, dissipation

        Returns:
            gamma_list[lst[flt]]: gamma values
            kernels[lst[np array]]: kernels normalized to first moment -1
            rAUC [lst[flt]]: AUCs calculated w/ riemann rule
            tAUC [lst[flt]]: AUCs calculated w/ trapezoidal rule

        '''
        N = net.ladder(5)
        g, k, rAUC, tAUC = N.get_random_kernels(RATE_FACTOR = RATE_FACTOR, s= s)
        return g, k, rAUC, tAUC


# SECTION: DATA ANALYSIS

    def r_value (self, x,y):

        """ Get the r value (correlation) of two lists

        Input:
            x [list]: list 1
            y [list]: list 2

        Output: r_value [float]: statistical r value of x, y
        """

        slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
        return r_value


    def small_Amax_f3_p1(self,Am,plot='no',t1=1,t2=9,average=0):

        """ Create a curve of synapse response to signal amplitude and get stats

        Input:
            Am [float]: Maximum aplitude at which performance is desired
            plot [str]: whether to create a visual plot
            t1 [float]: time for upswing
            t2 [float]: time of downswing (slow_time = t1 + t2)
            average [float]: mean value of sawtooth signal

        Output:
            Aminplus [float]: minimum positive value at which synapse response
                is linear with amplitude (divided by Amax)
            intercept [float]: y intercept of the linear portion of the synapse
                response vs amplitude graph. reflects offset caused by nonzero
                background signal
        """

        # initialize lists
        ds_list = []
        A_list = []

        theta = Am / t2 #pick a theta that will stop range of performance at Am

        for i in np.linspace(0,3 * Am / 4, int(4 * Am), endpoint = False): # sample a range of amplitudes
            # record A, ds relative to Am
            A_list.append(i/Am)
            deviation = i / 2 # deviate half the amplitude above and below the average value
            # compute and append the synapse response
            ds = self.compute_period_weight_update(average - deviation, average + deviation, fast_time = t1, slow_time = t1+t2, nonlinearity_threshold = theta, nonlinearity_style = "linear_threshold")
            ds_list.append(ds/Am)

        # get and record the y intercept and Amin from a 3 part linear fit
        intercept, Aminplus = self.piecewise_failure_stats(A_list,ds_list,3)

        if plot == 'yes':
            plt.plot(A_list,ds_list,label = Am)
            plt.show()

        return Aminplus, intercept

    def new_RAND_resp_curve(self, Am, AUC,t1=5,t2=15,average=0):
        """
        AN EDITED VERSION OF SMALL_AMAX_F3_P1
        Create a curve of synapse response to signal amplitude and get stats

        Input:
            Am [float]: Maximum aplitude at which performance is desired
            plot [str]: whether to create a visual plot
            t1 [float]: time for upswing
            t2 [float]: time of downswing (slow_time = t1 + t2)
            average [float]: mean value of sawtooth signal

        Output:
            Aminplus [float]: minimum positive value at which synapse response
                is linear with amplitude (divided by Amax)
            intercept [float]: y intercept of the linear portion of the synapse
                response vs amplitude graph. reflects offset caused by nonzero
                background signal
        """

        # initialize lists

        ds_list = []
        A_list = []


        theta = np.abs((-1 * Am / t2) + (AUC * (average + (Am / 2)))) #adaptively chosen theta

        for i in np.linspace(0, 1.25 * Am, 500, endpoint = False): # sample a range of amplitudes (500 data points)
            # record A, ds relative to Am
            A_list.append(i/Am)
            deviation = i / 2 # deviate half the amplitude above and below the average value
            # compute and append the synapse response
            ds = self.compute_period_weight_update(average - deviation, average + deviation, fast_time = t1, slow_time = t1+t2, nonlinearity_threshold = theta, nonlinearity_style = "linear_threshold")
            ds_list.append(ds/Am)

        return A_list, ds_list


    def piecewise_failure_stats(self,x,y,n=3):

        """ Get statistics after fitting linear segments to graph

        Input:
            x [list]: list of x values
            y [list]: list of y values
            n [int]: number of lines to fit to the data

        Output:
            intercept [float]: y intercept of 3rd section of the linear fit,
                corresponding to the linear response
            Amin [float]: left boundary of the 3rd section, corresponsing to the
                first value of amplitude which attains linear response
        """

        # convert lists to arrays
        x=np.array(x)
        y=np.array(y)

        # fit the piecewise representation
        my_pwlf = pwlf.PiecewiseLinFit(x, y)
        breaks = my_pwlf.fit(n) # n is number of pieces
        slopes = my_pwlf.calc_slopes() # also calculates intercepts stored at intercepts
        intercepts = my_pwlf.intercepts

        intercept = intercepts[2]
        # get left side of linear range if there is one
        if slopes[2] < 1.25 and slopes[2] > 0.75:
            Amin = breaks[2]
        else:
            Amin = 1
        print(f'braks = {breaks}')
        return intercept, Amin

    #### Analysis of Data from 6B

    def f4_error_analysis(self):

        """ Get data generated for figure 6B, calculate stats, error metrics"""

        bigdata = np.load('data_6B.npy', allow_pickle = True) #100, 50, 7 =  # [self.kernel, gamma, AUC, s, RATE, A_list, ds_list]

        data = []
        N = net.ladder(5)  #for diss
        for i in range(100):
            print(f'layer {i} of 100')
            for j in range(50):
                #get Aminrat
                Aminrat = 1
                Aminindex = 0
                l = int(4*len(bigdata[1][1][6])/5)
                for k, val in enumerate(bigdata[i][j][6][:l]):
                    if val > 0.05:
                        Aminrat = k / (4*l/5)    # there are 100 datat points going to 1.25Amax, so point 80 is Amax
                        Aminindex = k
                        break
                # get offset of postiive portion of response before Amax
                slope, intercept, r_val, p_val, std_err = stats.linregress(bigdata[i][j][5][Aminindex:l], bigdata[i][j][6][Aminindex:l])
                # get dissipation
                diss = N.stable_dissipation(bigdata[i][j][1]) #function of gamma
                data.append([Aminrat, intercept, diss])
        np.save('6B_stats', data)
        return data


# SECTION: DATA GENERATION

#### 5C: Generate Data for Figure 5C on the midway UChicago Supercomputer

    def midway_expanded_Amax_f3_p2_cube(self,I,av):

        """
        Generate the 12_24 cube

        This data with the dollowing functions was used to create plots t1_12_24_int.pdf, testamax.pdf (fig. 3C),
        and ecoli_0_Amax_f3_p2_v2.pdf

        used in conjunction with the ecoli_cube_stacker.py file to create a large array of data
        """

        task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
        max_task_id = os.environ.get('SLURM_ARRAY_TASK_MAX')
        #task_id should go 0 to 49
        total = 5 #number of points to be generated per zero indexed script to get 1000 total
        self.kernel = self._initialize_uneven_ecoli_kernel(I)
        Adatacube = np.zeros((total,4)) #
        for counter in range(total): #
            t1 = (np.random.randint(1,25))/5
            t2 = (np.random.randint(1,250))/5
            Amin_ratio, intercept = self.small_Amax_f3_p1(100,plot='no',n=1,t1=t1,t2=t2,average=av) #this returns Amin/Amax for given amax
            Adatacube[counter] = [t1,t2,Amin_ratio,intercept]
        np.save(f'midway_expanded_ecoli_cube_12_24_{task_id}.npy',Adatacube)

##### 4C: Generate Data for Figure 4C on the midway UChicago Supercomputer

    def midway_zero_cube(self):
        task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
        max_task_id = os.environ.get('SLURM_ARRAY_TASK_MAX')
        av = 0 # the average signal
        I = 0 # AUC of ecoli kernel
        total = 10
        self.kernel = self._initialize_uneven_ecoli_kernel(I)
        Adatacube = np.zeros((total,4))
        for counter in range(total):
            t1 = (np.random.randint(1,50))/10
            t2 = (np.random.randint(1,250))/5
            Amin_ratio, intercept = self.small_Amax_f3_p1(100,plot='no',t1=t1,t2=t2,average=av) #this returns Amin/Amax for given amax
            Adatacube[counter] = [t1,t2,Amin_ratio,intercept]
        np.save(f'Z_cube_1_26_{task_id}.npy', Adatacube)

#### 6B: Generate Data for Figure 6B on the midway UChicago Supercomputer

    def midway_f3p4_v5(self):
            """ use a batch job to create 5000 response
            curves using random markov kernels"""

            # each of 100 nodes gets task id
            taskid = int(os.environ.get('SLURM_ARRAY_TASK_ID') )

            # randomly choose
            s = np.random.randint(5,250)
            RATE = np.random.randint(1,100)

            # get 50 kernels
            g, k, rAUC, tzAUC = self.get_RAND_kernels(RATE_FACTOR = RATE, s= s)

            datacube = []

            for i in range(50):
                self.kernel = k[i]
                gamma = g[i]
                AUC = rAUC[i]

                # make the plot
                A_list, ds_list = self.new_RAND_resp_curve(50, AUC, t1 = 3, t2 = 15, average = 10)

                datacube.append([self.kernel, gamma, AUC, s, RATE, A_list, ds_list])

            np.save(f'f3p4data{taskid}_v7.py', datacube)

#SECTION: RUNNING

Sim = Simulation()

#run desired functions here
