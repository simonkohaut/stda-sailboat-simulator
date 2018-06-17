#!/usr/bin/env python3
# -*- coding: utf-8 -*-
''' This module simulates a sailboat with 6 degrees of freedom. '''


###################################################################################################
# Standard import
from sys         import argv
from math        import pi, sin, cos, atan, atan2, sqrt, copysign, degrees, radians, modf
from collections import namedtuple
from time        import sleep, clock


###################################################################################################
# Library import
import numpy as np
import yaml
from scipy.integrate import ode


###################################################################################################
# Load simulation parameters
sim_params = open('sim_params_config.yaml')
param_dict = yaml.load(sim_params)

# Boat
BOAT_LENGTH                       = param_dict['boat']['length']
MASS                              = param_dict['boat']['mass']
HULL_SPEED                        = param_dict['boat']['hull_speed']
LATERAL_AREA                      = param_dict['boat']['lateral_area']
WATERLINE_AREA                    = param_dict['boat']['waterline_area']
HEIGHT_BOUYANCY                   = param_dict['boat']['height_bouyancy']
# MOI : moment of inertia
GEOMETRICAL_MOI_X                 = param_dict['boat']['geometrical_moi_x']
GEOMETRICAL_MOI_Y                 = param_dict['boat']['geometrical_moi_y']
MOI_X                             = param_dict['boat']['moi_x']
MOI_Y                             = param_dict['boat']['moi_y']
MOI_Z                             = param_dict['boat']['moi_z']
# COG : center of gravity
DISTANCE_COG_RUDDER               = param_dict['boat']['distance_cog_rudder']
DISTANCE_COG_SAIL_PRESSURE_POINT  = param_dict['boat']['distance_cog_sail_pressure_point']
DISTANCE_COG_KEEL_PRESSURE_POINT  = param_dict['boat']['distance_cog_keel_pressure_point']
DISTANCE_MAST_SAIL_PRESSURE_POINT = param_dict['boat']['distance_mast_sail_pressure_point']
# Sail
SAIL_LENGTH                       = param_dict['boat']['sail']['length']
SAIL_HEIGHT                       = param_dict['boat']['sail']['height']
SAIL_AREA                         = param_dict['boat']['sail']['area']
SAIL_STRETCHING                   = param_dict['boat']['sail']['stretching']
SAIL_PRESSURE_POINT_HEIGHT        = param_dict['boat']['sail']['pressure_point_height']
# Keel
KEEL_LENGTH                       = param_dict['boat']['keel']['length']
KEEL_HEIGHT                       = param_dict['boat']['keel']['height']
KEEL_STRETCHING                   = param_dict['boat']['keel']['stretching']
# Rudder
RUDDER_BLADE_AREA                 = param_dict['boat']['rudder']['area']
RUDDER_STRETCHING                 = param_dict['boat']['rudder']['stretching']
# Damping
ALONG_DAMPING                     = param_dict['boat']['along_damping']
TRANSVERSE_DAMPING                = param_dict['boat']['transverse_damping']
DAMPING_Z                         = param_dict['boat']['damping_z']
YAW_TIMECONSTANT                  = param_dict['boat']['yaw_timeconstant']
PITCH_DAMPING                     = param_dict['boat']['pitch_damping']
ROLL_DAMPING                      = param_dict['boat']['roll_damping']
# Physical constants
WATER_DENSITY                     = param_dict['environment']['water_density']
WATER_VISCOSITY                   = param_dict['environment']['water_viscosity']
AIR_VISCOSITY                     = param_dict['environment']['air_viscosity']
AIR_DENSITY                       = param_dict['environment']['air_density']
GRAVITY                           = param_dict['environment']['gravity']


###################################################################################################
# Invariants
# Rudder force
#RUDDER_FORCE_INVARIANT_X = -(WATER_DENSITY / 2) * RUDDER_BLADE_AREA
#RUDDER_FORCE_INVARIANT_Y = 2 * pi * (WATER_DENSITY / 2) * RUDDER_BLADE_AREA
# Wave impedance
WAVE_IMPEDANCE_INVARIANT = (WATER_DENSITY / 2) * LATERAL_AREA
# Hydrostatic force
HYDROSTATIC_EFF_X        = HEIGHT_BOUYANCY + (WATER_DENSITY / MASS) * GEOMETRICAL_MOI_X
HYDROSTATIC_EFF_Y        = HEIGHT_BOUYANCY + (WATER_DENSITY / MASS) * GEOMETRICAL_MOI_Y
HYDROSTATIC_INVARIANT_Z  =  - WATER_DENSITY * WATERLINE_AREA * GRAVITY
GRAVITY_FORCE            = MASS * GRAVITY
# Damping
DAMPING_INVARIANT_X      = -MASS / ALONG_DAMPING
DAMPING_INVARIANT_Y      = -MASS / TRANSVERSE_DAMPING
DAMPING_INVARIANT_Z      = -.5 * DAMPING_Z * sqrt(WATER_DENSITY * WATERLINE_AREA * GRAVITY * MASS)
DAMPING_INVARIANT_YAW    = -(MOI_Z / YAW_TIMECONSTANT)
DAMPING_INVARIANT_PITCH  = -2 * PITCH_DAMPING * sqrt(MOI_Y * MASS * GRAVITY * HYDROSTATIC_EFF_Y)
DAMPING_INVARIANT_ROLL   = -2 * ROLL_DAMPING * sqrt(MOI_X * MASS * GRAVITY * HYDROSTATIC_EFF_X)


###################################################################################################
# Structured data
# Environment
TrueWind         = namedtuple('TrueWind', 'x, y, strength, direction')
ApparentWind     = namedtuple('ApparentWind', 'x, y, angle, speed')
Wave             = namedtuple('Wave', 'length, direction, amplitude')
WaveVector       = namedtuple('WaveVector', 'x, y')
WaveInfluence    = namedtuple('WaveInfluence', 'height, gradient_x, gradient_y')
# Forces
RudderForce      = namedtuple('RudderForce', 'x, y')
LateralForce     = namedtuple('LateralForce', 'x, y')
SailForce        = namedtuple('SailForce', 'x, y')
HydrostaticForce = namedtuple('HydrostaticForce', 'x, y, z')
Damping          = namedtuple('Damping', 'x, y, z, yaw, pitch, roll')


###################################################################################################
# State description
POS_X,     POS_Y,      POS_Z,             \
ROLL,      PITCH,      YAW,               \
VEL_X,     VEL_Y,      VEL_Z,             \
ROLL_RATE, PITCH_RATE, YAW_RATE = range(12)

def initial_state():
    ''' Returns the initial state for a simulation.
        Consists of position, velocity, rotation and angular velocity. '''
    return np.array([0,
                     0,
                     0,
                     param_dict['simulator']['initial']['roll'],
                     param_dict['simulator']['initial']['pitch'],
                     param_dict['simulator']['initial']['yaw'],
                     param_dict['simulator']['initial']['vel_x'],
                     param_dict['simulator']['initial']['vel_y'],
                     param_dict['simulator']['initial']['vel_z'],
                     param_dict['simulator']['initial']['roll_rate'],
                     param_dict['simulator']['initial']['pitch_rate'],
                     param_dict['simulator']['initial']['yaw_rate']])


###################################################################################################
# Environment index description
SAIL_ANGLE, RUDDER_ANGLE, TRUE_WIND, WAVE = range(4)


###################################################################################################
# Force calculations
# pylint: enable = C0326
def sign(value):
    ''' Implements the sign function. 
    
    param value: The value to get the sign from

    return: The sign of value {-1, 0, 1}
    '''
    return copysign(1, value) if value != 0 else 0


def calculate_apparent_wind(yaw, vel_x, vel_y, true_wind):
    ''' Calculate the apparent wind on the boat. 

    param yaw:          The heading of the boat [radians]
    param vel_x:        The velocity along the x-axis [m/s]
    param vel_y:        The velocity along the y-axis [m/s]
    param true_wind:    The true wind directions

    return: The apparent wind on the boat
    '''
    transformed_x = true_wind.x * cos(yaw) + true_wind.y * sin(yaw)
    transformed_y = true_wind.x * -sin(yaw) + true_wind.y * cos(yaw)

    apparent_x = transformed_x - vel_x
    apparent_y = transformed_y - vel_y
    apparent_angle = atan2(-apparent_y, -apparent_x)
    apparent_speed = sqrt(apparent_x**2 + apparent_y**2)

    return ApparentWind(x=apparent_x,
                        y=apparent_y,
                        angle=apparent_angle,
                        speed=apparent_speed)


def calculate_sail_force(roll, wind, sail_angle):
    ''' Calculate the force that is applied to the sail. 
    
    param roll:         The roll angle of the boat [radians]
    param wind:         The apparent wind on the boat  
    param sail_angle:   The angle of the main sail [radians]

    return: The force applied on the sail by the wind
    '''
    # aoa : angle of attack
    aoa = wind.angle - sail_angle
    if aoa * sail_angle < 0:
        aoa = 0
    # eff_aoa : effective angle of attack
    eff_aoa = aoa
    if aoa < -pi / 2:
        eff_aoa = pi + aoa
    elif aoa > pi / 2:
        eff_aoa = -pi + aoa

    pressure = (AIR_DENSITY / 2) * wind.speed**2 * cos(roll * cos(sail_angle))**2

    friction = 3.55 * sqrt(AIR_VISCOSITY / (wind.speed * SAIL_LENGTH)) \
               if wind.speed != 0                                      \
               else 0
    
    separation = 1-np.exp(-((abs(eff_aoa))/(pi/180*25))**2)
    
    propulsion = (2 * pi * eff_aoa * sin(wind.angle)                                       \
                  -(friction + (4 * pi * eff_aoa**2 * separation) / SAIL_STRETCHING) * cos(wind.angle)) \
                  * SAIL_AREA * pressure

    transverse_force = (-2 * pi * eff_aoa * cos(wind.angle)                                      \
                        -(friction + (4 * pi * eff_aoa**2 * separation) / SAIL_STRETCHING) * sin(wind.angle)) \
                        * SAIL_AREA * pressure

    separated_propulsion = sign(aoa) * pressure * SAIL_AREA * sin(aoa)**2 * sin(sail_angle)
    separated_transverse_force = -sign(aoa) * pressure * SAIL_AREA * sin(aoa)**2 * cos(sail_angle)
    
    return SailForce(
        x=(1 - separation) * propulsion + separation * separated_propulsion,
        y=(1 - separation) * transverse_force + separation * separated_transverse_force)


def calculate_lateral_force(vel_x, vel_y, roll, speed):
    ''' Calculate the lateral force. 
    
    param vel_x:        The velocity along the x-axis   [m/s]
    param vel_y:        The velocity along the y-axis   [m/s]
    param roll:         The roll angle of the boat      [radians]
    param speed:        The total speed of the boat     [m/s]

    return: The force applied to the lateral plane of the boat
    '''
    pressure = (WATER_DENSITY / 2) * speed**2 * cos(roll)**2

    friction = 2.66 * sqrt(WATER_VISCOSITY / (speed * KEEL_LENGTH)) \
               if speed != 0                                        \
               else 0

    #     aoa :           angle of attack
    # eff_aoa : effective angle of attack
    eff_aoa = aoa = atan2(vel_y, vel_x)
    if aoa < -pi / 2:
        eff_aoa = pi + aoa
    elif aoa > pi / 2:
        eff_aoa = -pi + aoa
    
    separation = 1-np.exp(-((abs(eff_aoa))/(pi/180*25))**2)

    # Identical calculation for x and y
    tmp = -(friction + (4 * pi * eff_aoa**2 * separation) / KEEL_STRETCHING)
    
    separated_transverse_force = -sign(aoa) * pressure * SAIL_AREA * sin(aoa)**2
    
    return LateralForce(
        x=(1 - separation) * (tmp * cos(aoa) + 2 * pi * eff_aoa * sin(aoa)) * pressure * LATERAL_AREA,
        y=(1 - separation) * (tmp * sin(aoa) - 2 * pi * eff_aoa * cos(aoa)) * pressure * LATERAL_AREA + separation * separated_transverse_force)\
            , separation


def calculate_rudder_force(speed, rudder_angle):
    ''' Calculate the force that is applied to the rudder. 
   
    param speed:        The total speed of the boat [m/s]
    param rudder_angle: The angle of the rudder     [radians]

    return: The force applied to the rudder of the boat
    '''
    pressure = (WATER_DENSITY / 2) * speed**2
    return RudderForce(
        x=-(((4 * pi) / RUDDER_STRETCHING) * rudder_angle**2) * pressure * RUDDER_BLADE_AREA,
        y=2 * pi * pressure * RUDDER_BLADE_AREA * rudder_angle)


def calculate_wave_impedance(vel_x, speed):
    ''' Calculate the wave impedance. 
    
    param vel_x: The velocity along the x-axis  [m/s]    
    param speed: The total speed of the boat    [m/s]

    return: The force applied to the rudder of the boat
    '''
    return -sign(vel_x) * speed**2 * (speed / HULL_SPEED)**2 * WAVE_IMPEDANCE_INVARIANT


def calculate_wave_influence(pos_x, pos_y, yaw, wave, time):
    ''' Calculate how the waves influence the boat. 
    
    param pos_x:    The boats position on the x-axis        [m]  
    param pos_y:    The boats position on the y-axis        [m]
    param yaw:      The heading of the boat                 [radians]        
    param wave:     The direction and length of the waves
    param time:     The simulation time                     [s]

    return: The influence of the waves on the boat
    '''
    frequency = sqrt((2 * pi * GRAVITY) / wave.length)

    k = WaveVector(x=2 * pi / wave.length * cos(wave.direction),
                   y=2 * pi / wave.length * sin(wave.direction))

    factor = -wave.amplitude * cos(frequency * time - k.x * pos_x - k.y * pos_y)
    gradient_x = k.x * factor
    gradient_y = k.y * factor

    return WaveInfluence(
        height=wave.amplitude * sin(frequency * time - k.x * pos_x - k.y * pos_y),
        gradient_x=gradient_x * cos(yaw) + gradient_y * sin(yaw),
        gradient_y=gradient_y * cos(yaw) - gradient_x * sin(yaw))


def calculate_hydrostatic_force(pos_z, roll, pitch, wave_influence):
    ''' Calculate the hydrostatic force. 
    
    param pos_z:            The boats position on the z-axis        [m]  
    param roll:             The roll angle of the boat              [radians]
    param pitch:            The pitch angle of the boat             [radians]
    param wave_influence:   The influence of the waves on the boat     

    return: The force applied on the boat by the waves
    '''
    force = HYDROSTATIC_INVARIANT_Z * (pos_z - wave_influence.height) + GRAVITY_FORCE
    
    return HydrostaticForce(
        x=force * wave_influence.gradient_x,
        y=force * wave_influence.gradient_y,
        z=force), \
            HYDROSTATIC_EFF_Y * sin(pitch + atan(wave_influence.gradient_x)), \
            HYDROSTATIC_EFF_X * -sin(roll - atan(wave_influence.gradient_y)),


def calculate_damping(vel_x, vel_y, vel_z, roll_rate, pitch_rate, yaw_rate):
    ''' Calculate the damping. 
    
    param vel_x:        The velocity along the x-axis           [m/s]  
    param vel_y:        The velocity along the y-axis           [m/s]  
    param vel_z:        The velocity along the z-axis           [m/s]  
    param roll_rate:    The rate of change to the roll angle    [radians/s]
    param pitch_rate:   The rate of change to the pitch angle   [radians/s]
    param yaw_rate:     The rate of change to the yaw angle     [radians/s]

    return: The amount of damping applied to the boat
    '''
    return Damping(
        x=DAMPING_INVARIANT_X * vel_x,
        y=DAMPING_INVARIANT_Y * vel_y,
        z=DAMPING_INVARIANT_Z * vel_z,
        roll=DAMPING_INVARIANT_ROLL * roll_rate,
        pitch=DAMPING_INVARIANT_PITCH * pitch_rate,
        yaw=DAMPING_INVARIANT_YAW * yaw_rate)

RUDDER_STATE, SAIL_STATE = 12, 13
actor_dynamics = True

DISTANCE_COG_KEEL_MIDDLE = DISTANCE_COG_KEEL_PRESSURE_POINT - .7

def solve(time, boat):
    t1 = clock()
    ''' Solve the ode for the given state, time and environment. '''
    # State unpacking
    # pylint: disable = C0326
    pos_x,     pos_y,      pos_z    = boat[POS_X     : POS_Z    + 1]
    roll,      pitch,      yaw      = boat[ROLL      : YAW      + 1]
    vel_x,     vel_y,      vel_z    = boat[VEL_X     : VEL_Z    + 1]
    roll_rate, pitch_rate, yaw_rate = boat[ROLL_RATE : YAW_RATE + 1]
    if actor_dynamics:
        rudder_angle, sail_angle = boat[RUDDER_STATE: SAIL_STATE + 1]
    # environment unpacking
        sail_angle_reference   = environment[SAIL_ANGLE]
        rudder_angle_reference = environment[RUDDER_ANGLE]
    else:
        sail_angle   = environment[SAIL_ANGLE]
        rudder_angle = environment[RUDDER_ANGLE]
    
    wave         = environment[WAVE]
    true_wind    = environment[TRUE_WIND]
    
    # For force calulations needed values
    speed             = sqrt(vel_x**2 + vel_y**2)# + vel_z**2)
    wave_influence    = calculate_wave_influence(pos_x, pos_y, yaw, wave, time)
    apparent_wind     = calculate_apparent_wind(yaw, vel_x, vel_y, true_wind)
    
    # sail angle is determined from rope length and wind direction
    true_sail_angle = np.sign(apparent_wind.angle) * abs(sail_angle)
    
    # Force calculation
    damping           = calculate_damping(vel_x, vel_y, vel_z, roll_rate, pitch_rate, yaw_rate)
    hydrostatic_force, x_hs, y_hs = calculate_hydrostatic_force(pos_z, roll, pitch, wave_influence)
    
    wave_impedance    = calculate_wave_impedance(vel_x, speed)
    rudder_force      = calculate_rudder_force(speed, rudder_angle)
    lateral_force, lateral_separation     = calculate_lateral_force(vel_x, vel_y, roll, speed)
    #lateral_separation = 0
    sail_force        = calculate_sail_force(roll, apparent_wind, true_sail_angle)

    # Calculate changes
    delta_pos_x = vel_x * cos(yaw) - vel_y * sin(yaw)
    delta_pos_y = vel_y * cos(yaw) + vel_x * sin(yaw)
    delta_pos_z = vel_z
    delta_roll  = roll_rate
    delta_pitch = pitch_rate * cos(roll) - yaw_rate * sin(roll)
    delta_yaw   = yaw_rate * cos(roll) + pitch_rate * sin(roll)
    
    delta_vel_x = delta_yaw * vel_y + (sail_force.x + lateral_force.x + rudder_force.x + damping.x + wave_impedance + hydrostatic_force.x) / MASS
     
    delta_vel_y = -delta_yaw * vel_x + \
                  ((sail_force.y + lateral_force.y + rudder_force.y) * cos(roll) + \
                  hydrostatic_force.y + \
                   damping.y) / MASS

    delta_vel_z = ((sail_force.y + lateral_force.y + rudder_force.y) * sin(roll) + \
                  hydrostatic_force.z - GRAVITY_FORCE + damping.z) / MASS
                   #MASS * GRAVITY + damping.z) / MASS
    

    delta_roll_rate  = (hydrostatic_force.z * y_hs
                        - sail_force.y * SAIL_PRESSURE_POINT_HEIGHT
                        + damping.roll) / MOI_X

    delta_pitch_rate = (sail_force.x * SAIL_PRESSURE_POINT_HEIGHT
                        - hydrostatic_force.z * x_hs * cos(roll)
                        + damping.pitch
                        - (MOI_X - MOI_Z) * roll_rate * yaw_rate) / MOI_Y
    
    delta_yaw_rate   = (damping.yaw
                        #+ hydrostatic_force.z * hydrostatic_force.x * sin(roll)
                        - rudder_force.y * DISTANCE_COG_RUDDER
                        + sail_force.y * DISTANCE_COG_SAIL_PRESSURE_POINT
                        + sail_force.x * sin(true_sail_angle) * DISTANCE_MAST_SAIL_PRESSURE_POINT
                        + lateral_force.y * (DISTANCE_COG_KEEL_PRESSURE_POINT * (1-lateral_separation)\
                                            + DISTANCE_COG_KEEL_MIDDLE * lateral_separation))/ MOI_Z
    
    if actor_dynamics:
        # actor dynamics
        delta_rudder = - 2 * (rudder_angle - rudder_angle_reference)
        max_rudder_speed = pi/30
        #if delta_rudder > max_rudder_speed:
            #print delta_rudder, max_rudder_speed
        delta_rudder = np.clip(delta_rudder, -max_rudder_speed, max_rudder_speed)
        
        
        delta_sail = - .1 * (sail_angle - sail_angle_reference)
        max_sail_speed = pi/10
        delta_sail = np.clip(delta_sail, -max_sail_speed, max_sail_speed)
                           
                           
    
    delta = np.array(
        [delta_pos_x,     delta_pos_y,      delta_pos_z,
         delta_roll,      delta_pitch,      delta_yaw,
         delta_vel_x,     delta_vel_y,      delta_vel_z,
         delta_roll_rate, delta_pitch_rate, delta_yaw_rate])
    if actor_dynamics:
        delta = np.concatenate((delta, [
         delta_rudder, delta_sail]))
    return delta


###################################################################################################
# Simulation parameters
STEPSIZE  = param_dict['simulator']['stepper']['stepsize']
CLOCKRATE = param_dict['simulator']['stepper']['clockrate']


###################################################################################################
# Simulation state
INITIAL_WIND_STRENGTH  = param_dict['simulator']['initial']['wind_strength']
INITIAL_WIND_DIRECTION = param_dict['simulator']['initial']['wind_direction']
INITIAL_WAVE_DIRECTION = param_dict['simulator']['initial']['wave_direction']
INITIAL_WAVE_LENGTH    = param_dict['simulator']['initial']['wave_length']
INITIAL_WAVE_AMPLITUDE = param_dict['simulator']['initial']['wave_amplitude']
INITIAL_SAIL_ANGLE     = param_dict['simulator']['initial']['sail_angle']
INITIAL_RUDDER_ANGLE   = param_dict['simulator']['initial']['rudder_angle']
    
state       = initial_state()
environment = [INITIAL_SAIL_ANGLE,
               INITIAL_RUDDER_ANGLE,
               TrueWind(INITIAL_WIND_STRENGTH * cos(radians(INITIAL_WIND_DIRECTION)),
                        INITIAL_WIND_STRENGTH * sin(radians(INITIAL_WIND_DIRECTION)),
                        INITIAL_WIND_STRENGTH,
                        INITIAL_WIND_DIRECTION),
               Wave(direction=INITIAL_WAVE_DIRECTION,
                    length=INITIAL_WAVE_LENGTH,
                    amplitude=INITIAL_WAVE_AMPLITUDE)]


###################################################################################################
# Simulate for a given timestep 
def step(event):
    ''' Evaluate the new state. '''
    now = rp.Time.now().to_sec()
    end = now + 0.1
    global state # pylint: disable = W0603, C0103
    state = ode(solve).set_integrator('dopri5', nsteps=1000).set_initial_value(state, 0).integrate(STEPSIZE)

    if state[YAW] < -2 * pi:
        state[YAW] += 2 * pi
    elif state[YAW] > 2 * pi:
        state[YAW] -= 2 * pi
