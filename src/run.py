
from heading_controller import heading_controller
from simulation import *
from numpy import *
import matplotlib.pyplot as plt
from sail_angle import sail_angle
from time import clock

deq = solve

def main():
    
    save=True
    #save=False
    
    scenario_1(save=save)
    scenario_0(save=save)
    
    plt.show()

def deq_sparse(time, state):
    # ignore pitch and heave oscillations (not really needed)
    diff = deq(time, state)
    diff[VEL_Z] = 0
    diff[PITCH_RATE] = 0
    diff[ROLL_RATE] = 0
    return diff
    

    
def init_data_arrays(n_states, N_steps, x0):
    x = zeros((n_states, N_steps+1))
    r = zeros(N_steps+1)
    sail = zeros(N_steps+1)
    t = zeros(N_steps+1)    
    x [:, 0] = x0
    
    ref_heading = zeros(N_steps+1)
    return x, t, r, sail, ref_heading
        
def init_integrator(x0, sampletime, sparse=False):
    # init integrator
    fun = deq if not sparse else deq_sparse
    integrator = ode(fun).set_integrator('dopri5')    
    integrator.set_initial_value(x0, 0)
    # init heading controller
    controller = heading_controller(sampletime, max_rudder_angle=15*pi/180)
    controller.calculate_controller_params(YAW_TIMECONSTANT)
    
    return integrator, controller

def scenario_1(save=False):
    n_states = 12
    if actor_dynamics:
        n_states = 14
    #def set_environment():
    ## environment
    #wind = TrueWind(0, 3, 3, pi/2)
    wind = TrueWind(0, 5, 5, pi/2)
    #wind = TrueWind(0, 2, 2, pi/2) 
    #wind = TrueWind(0, 0, 0, pi/2) 
    environment[SAIL_ANGLE] = 0. /180 * pi
    environment[RUDDER_ANGLE] = 0
    environment[WAVE] = Wave(50, 0, 0)
    sim_wave = False
    if sim_wave:
        environment[WAVE] = Wave(length=100., amplitude=.5, direction=0)
        append = "_wave"
    else:
        
        append = ""
    environment[TRUE_WIND] = wind
    
    # simulation params
    t_end = 150.
    #t_end = 57.
    sampletime = .3
    sail_sampletime = 2.
    N_steps = int(t_end / sampletime)
    # initial values
    x0 = zeros(n_states)
    x0[VEL_X] = 0.
    
    if actor_dynamics:
        x0[SAIL_STATE] = 48*pi/180
    
    x, t, r, sail, ref_heading = init_data_arrays(n_states, N_steps, x0)
    
    if True:
        x2, t2, r2, sail2, ref_heading = init_data_arrays(n_states, N_steps, x0)
    
    
    integrator, controller = init_integrator(x0, sampletime)
    integrator2 = simple_integrator(deq_sparse)
    integrator2.set_initial_value(x0, 0)
    
    #controller2 = heading_controller(sampletime)
    ##controller2.KP = 0
    ##controller2.KD = 0
    ##controller2.KI = controller2.KI /100
    #controller2.calculate_controller_params(YAW_TIMECONSTANT, Q=diag([1E-1, 1, 5]), r=ones((1,1))*1000)

    # reference trajectory heading    
    
    #ref_heading[:int(30/sampletime)] = -.2*pi*0
    
    
    ref_heading[int(40/sampletime):] = 1.2*pi
    
    #ref_heading[int(30/sampletime):int(33/sampletime)] = 0.8*pi
    #ref_heading[int(30/sampletime):] = .9*pi
    
    ref_heading[int(90/sampletime) :] = .35 * pi
    
    #ref_heading = smooth_reference(ref_heading, int(5/sampletime))
    
    def set_predefined_sail_angle():
        predefined_sail_angle = ones(N_steps + 1)
        predefined_sail_angle[:int(30/sampletime)] = 48*pi/180
        #predefined_sail_angle[:int(10/sampletime)] = -65*pi/180
        predefined_sail_angle[int(30/sampletime):] = 33*pi/180
        predefined_sail_angle[int(30/sampletime):int(50/sampletime)] = 48*pi/180
        predefined_sail_angle[int(48/sampletime):int(55/sampletime)] = 41*pi/180
        
        return predefined_sail_angle
    
    def set_predefined_sail_angle2():
        predefined_sail_angle = ones(N_steps + 1)
        predefined_sail_angle[:int(30/sampletime)] = 48*pi/180
        #predefined_sail_angle[:int(10/sampletime)] = -65*pi/180
        #predefined_sail_angle[int(30/sampletime):] = 33*pi/180
        predefined_sail_angle[int(30/sampletime):int(50/sampletime)] = 48*pi/180
        predefined_sail_angle[int(58/sampletime):] = 41*pi/180
        
        return predefined_sail_angle
    
    #sail_angle = set_predefined_sail_angle2()
    sail_angle = None
    
    x, t, separation, keel, sail_force, sail, r = simulate(N_steps, x, t, r, sail, environment, controller, 
                                                            integrator, sampletime, sail_sampletime, ref_heading, wind, sail_angle)
    x2, t2, separation2, keel2, sail_force2, sail2, r2 = simulate(N_steps, x2, t2, r2, sail, environment, controller, 
                                                            integrator2, sampletime, sail_sampletime, ref_heading, wind, sail_angle)
    
    #comp_route(x, x2)
    
    
    plots_manoevers(t, x, r, sail, ref_heading, save, sail_force=sail_force, keel_force=keel, separation=separation, wind=wind, append=append)


def smooth_reference(ref_heading, n_filter=5):
    ################ potential smoothing of heading reference
    smoothed = ref_heading.copy()
    N=ref_heading.shape[0]
    for i in range(N):
        ind_low = max(0, i-n_filter/2)
        ind_high = min(N, i+(n_filter-n_filter/2))
        smoothed[i] = np.mean(ref_heading[ind_low:ind_high])
        
    return smoothed

def comp_route(x, x2):
    fig_traj, ax_traj = plot_series(x[POS_X, :], x[POS_Y, :], xlabel='$x$ / m', ylabel='$y$ / m')
    fig_traj, ax_traj = plot_series(x2[POS_X, :], x2[POS_Y, :], xlabel='$x$ / m', ylabel='$y$ / m', fig=fig_traj, ax=ax_traj)
    
    

def plots_manoevers(t, x, r, sail, ref_heading, save, sail_force=None, keel_force=None, separation=None, wind=None, append=""):
    # plots
    
    speeds = [x[VEL_X, :], x[VEL_Y, :], x[VEL_Z, :]]
    speed_axlabels = ['Time / s', 'Speed / m/s']
    speed_labels = ['Forward', 'Leeway', 'Vertical']
    
    fig_speed, tmp = plot_time_fig(time=t, data=speeds, labels=speed_labels, ax_labels=speed_axlabels)
    
    angles = [x[ROLL, :], x[PITCH, :]]#, x[YAW, :]]
    angles_labels = ['Roll', 'Pitch']#, 'Heading']
    axlabels = ['Time / s', 'Angles / degree']
    scale = 180/pi
    
    fig_ang, tmp = plot_time_fig(time=t, data=angles, labels=angles_labels, ax_labels=axlabels, scale=scale)
    
    
    fig_traj, ax_traj = plot_series(x[POS_X, :], x[POS_Y, :], xlabel='$x_g$ / m', ylabel='$y_g$ / m')
    plot_arrows(x[POS_X, :], x[POS_Y, :], x[YAW, :], fig=fig_traj, ax=ax_traj, color='red')
    ax_traj.plot(x[POS_X, 0], x[POS_Y, 0], 'xb', markersize=10)
    ax_traj.arrow(0, -25, wind.x, wind.y, head_width=1.5, head_length=3, width=.5, fc='green', ec='green')
    #ax_traj.legend(['route', 'start', 'wind'])
    
    # heading plot
    heading_data = [x[YAW, :] + 2*pi, ref_heading, arctan2(x[VEL_Y, :], x[VEL_X, :])]
    heading_labels = ['Heading', 'Reference', 'Drift']
    fig_heading, ax_heading = plot_time_fig(time=t, data=heading_data, labels=heading_labels, ax_labels=axlabels, scale=scale, ticks='ang')
    
    
    if save: 
        fig_heading.savefig('figs/heading'+append+'.eps')
        fig_ang.savefig('figs/angles'+append+'.eps')
        fig_speed.savefig('figs/speeds'+append+'.eps')
        fig_traj.savefig('figs/trajectories'+append+'.eps')
        
    

def plot_time_fig(time, data, labels, ax_labels, scale=1., ticks=None):
    fig, ax = None, None
    for i, data_i in enumerate(data):
        fig, ax = plot_series(time, data_i*scale, label=labels[i], xlabel=ax_labels[0], ylabel=ax_labels[1], fig=fig, ax=ax, legend=True)
        
    if ticks is not None:
        if ticks == 'ang':
            ax.set_yticks(np.arange(9)*45)
    return fig, ax

def plot_old():
    
    fig_vx, tmp = plot_series(t, x[VEL_X, :], xlabel='time /s', ylabel='speed $x$ / m/s')
    fig_vy, tmp = plot_series(t, x[VEL_Y, :], xlabel='time /s', ylabel='speed $y$ / m/s')
    fig_heading, ax_heading = plot_series(t, (x[YAW, :]/pi*180) % 360, xlabel='time /s', ylabel='heading / deg')
    
    drift = drift = arctan2(x[VEL_Y, :], x[VEL_X, :])
    fig_heading, ax_heading = plot_series(t, ref_heading/pi*180, xlabel='time /s', ylabel='heading / deg', fig=fig_heading, ax=ax_heading)
    fig_heading, ax_heading = plot_series(t, drift/pi*180, xlabel='time /s', ylabel='heading / deg', fig=fig_heading, ax=ax_heading)
    
    
    fig_r, ax_r = plot_series(t, r/pi*180, xlabel='time /s', ylabel='rudder / deg')
    if actor_dynamics:
        fig_r, ax_r = plot_series(t, x[12,:]/pi*180, xlabel='time /s', ylabel='rudder / deg', fig=fig_r, ax=ax_r)
    #plot_series(t, x[ROLL,:]/pi*180, xlabel='time /s', ylabel='rudder / deg', fig=fig_r, ax=ax_r)
    fig_sail, ax_sail = plot_series(t, sail/pi*180, xlabel='time /s', ylabel='sail angle / deg')
    if actor_dynamics:
        plot_series(t, x[13,:]/pi*180, xlabel='time /s', ylabel='sail angle / deg', fig=fig_sail, ax=ax_sail)
    fig_traj, ax_traj = plot_series(x[POS_X, :], x[POS_Y, :], xlabel='$x_g$ / m', ylabel='$y_g$ / m')
    
    plot_arrows(x[POS_X, :], x[POS_Y, :], x[YAW, :], fig=fig_traj, ax=ax_traj, color='red')
    #dir = arctan2(x[VEL_Y, :], x[VEL_X, :]) + x[YAW, :]
    #plot_arrows(x[POS_X, :], x[POS_Y, :], dir, fig=fig_traj, ax=ax_traj, color='k')
    fig_pitch, tmp = plot_series(t, x[PITCH, :]/pi*180, xlabel='time /s', ylabel='pitch in deg')
    fig_roll, tmp = plot_series(t, x[ROLL, :]/pi*180, xlabel='time /s', ylabel='roll in deg')
    
    if sail_force is not None:
        plot_series(t, sail_force[1, :], ylabel='sailforce y')
        plot_series(t, keel_force[1, :], ylabel='keelforce y')
        plot_series(t, separation, ylabel='separation')
        
    if save: 
        fig_heading.savefig('figs/heading.eps')
        fig_vx.savefig('figs/speed.eps')
        fig_vy.savefig('figs/leeway_speed.eps')
        fig_r.savefig('figs/rudder.eps')
        fig_sail.savefig('figs/sail.eps')
        fig_traj.savefig('figs/trajectories.eps')
        
        fig_pitch.savefig('figs/pitch.eps')
        fig_roll.savefig('figs/roll.eps')
        
        
        
def scenario_0(save=False):
    n_states = 12
    if actor_dynamics:
        n_states = 14
    ## environment
    wind = TrueWind(0, 6, 6, pi/2) 
    #wind = TrueWind(0, 0, 0, pi/2) 
    environment[SAIL_ANGLE] = 0. /180 * pi
    environment[RUDDER_ANGLE] = 0
    #environment[WAVE] = Wave(50, 0, 1)
    environment[WAVE] = Wave(50, 0, 0)
    environment[TRUE_WIND] = wind
    
    # simulation params
    t_end = 3.
    #t_end = 57.
    sampletime = .01
    sail_sampletime = 5.
    N_steps = int(t_end / sampletime)
    # initial values
    x0 = zeros(n_states)
    x0[VEL_X] = 0.
    if actor_dynamics:
        x0[SAIL_STATE] = 48*pi/180
    
    x, t, r, sail, ref_heading = init_data_arrays(n_states, N_steps, x0)
    integrator, controller = init_integrator(x0, sampletime)
    sail_angle = np.ones(N_steps) * 48*pi/180 # set_predefined_sail_angle()
    
    x, t, separation, keel, sail_force, sail, r = simulate(N_steps, x, t, r, sail, environment, 
                                                           controller, integrator, sampletime, sail_sampletime, ref_heading, wind, sail_angle)
    
    
    fig, tmp = plot_leeways(t, separation, x[VEL_X], x[VEL_Y])
    if save:
        fig.savefig('figs/leeways.eps')

    
def plot_leeways(t, separation, vx, vy):
    ax, fig = None, None
    fig, ax = plot_series(t, vx, xlabel='Time /s', label='Forward speed in $m/s$', ax=ax, fig=fig)
    fig, ax = plot_series(t, vy, xlabel='Time /s', label='Leeway in $m/s$', ax=ax, fig=fig)
    fig, ax = plot_series(t, separation, xlabel='Time /s', label='Flow separation factor', ax=ax, fig=fig)
    
    #ax.grid(True)
    ax.legend()
    ax.set_xlim([0, 3])
    ax.set_ylim([0, 1])
    return fig, ax
    
def simulate(N_steps, x, t, r, sail, environment, controller, integrator, sampletime, sail_sampletime, ref_heading, wind=None, predefined_sail_angle=None):
    t1 = clock()
    for i in range(N_steps):
        
        # rudder_angle calculation
        speed = sqrt(x[VEL_X, i]**2 + x[VEL_Y, i]**2) 
        drift = arctan2(x[VEL_Y, i], x[VEL_X, i])
        environment[RUDDER_ANGLE] = controller.controll(ref_heading[i], x[YAW, i], x[YAW_RATE, i], speed, x[ROLL, i], drift_angle=drift)
        #if t[i] < 80:
        r[i] = environment[RUDDER_ANGLE]
        #else:
            #r[i] = controller2.controll(ref_heading[i], x[YAW, i], x[YAW_RATE, i], speed, x[ROLL, i])#-.3 /180 *pi
            #environment[RUDDER_ANGLE] = r[i]
            
        if not predefined_sail_angle is None:
            sail[i] = predefined_sail_angle[i]
            environment[SAIL_ANGLE] = sail[i]
        else:
            if not i%int(sail_sampletime/sampletime):
                apparent_wind = calculate_apparent_wind(x[YAW, i], x[VEL_X, i], x[VEL_Y, i], wind)        
                environment[SAIL_ANGLE] = sail_angle(apparent_wind.angle, apparent_wind.speed, SAIL_STRETCHING)
                
                #print apparent_wind.angle/pi*180, wind.direction/pi*180, x[VEL_X, i], x[VEL_Y, i], wind.strength, apparent_wind.speed, x[YAW, i]/pi*180
                
                sail[i] = environment[SAIL_ANGLE]    
                
            else:
                sail[i] = sail[i-1]
            
    

        integrator.integrate(integrator.t+sampletime)
        t[i+1] = integrator.t
        x[:, i+1] = integrator.y
        #print 'time', t[i]
    print 'computation time', clock()-t1
    # forces
    sail_force = np.zeros((2, x.shape[1]))
    keel = np.zeros((2, x.shape[1]))
    separation = np.zeros((x.shape[1]))
    for i in range(x.shape[1]):
        apparent_wind = calculate_apparent_wind(x[YAW, i], x[VEL_X, i], x[VEL_Y, i], wind)
        force = calculate_sail_force(x[ROLL, i], apparent_wind, np.sign(apparent_wind.angle)*sail[i])
        sail_force[:, i] = [force.x, force.y]
        speed = np.sqrt(x[VEL_X, i]**2+ x[VEL_Y, i]**2)
        lateral_force, lateral_separation     = calculate_lateral_force(x[VEL_X, i], x[VEL_Y, i], x[YAW, i], speed)
        keel[:, i] = [lateral_force.x, lateral_force.y]
        
        separation[i] = lateral_separation
        
    return x, t, separation, keel, sail_force, sail, r

class simple_integrator(object):
    
    def __init__(self, deq):
        
        self.t = 0
        self.deq = deq
        self.max_sample_time = .01
        pass
    
    def set_initial_value(self, x0, t0):
        
        self.y = x0
        self.t = t0
        
    def integrate(self, end_time):
        while end_time - self.t > 1.1 * self.max_sample_time:
            self.integrate(self.t + self.max_sample_time)
            
        
        
        self.y += self.deq(self.t, self.y) * (end_time - self.t)
        self.t = end_time
    
def plot_arrows(x, y, directions, fig=None, ax=None, color=None):
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(N_subplot, 1, n_subplot)
    if color is None:
        color = 'k'
    
    length = mean(abs(x[1:]-x[:-1]))
    #length = .2
    
    for i in range(x.shape[0]):
        ax.arrow(x[i], y[i], length*cos(directions[i]), length*sin(directions[i]), head_width=.05, head_length=.1, fc=color, ec=color)
    

    
def plot_series(x, y, fig=None, ax=None, N_subplot=1, n_subplot=1, title=None, xlabel=None, ylabel=None, label=None, legend=False):
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(N_subplot, 1, n_subplot)
    ax.plot(x, y, label=label)
    if not xlabel is None:
        ax.set_xlabel(xlabel)
    if not ylabel is None:
        ax.set_ylabel(ylabel)
    ax.grid(True)
    
    if legend:
        ax.legend()
    return fig, ax
    
if __name__ == "__main__":
    main()
