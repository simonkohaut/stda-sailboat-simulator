from numpy import *

def sail_angle(wind_angle, wind_speed, sail_stretching, limit_wind_speed=6, stall_deg=14.):
    
    #opt_aoa = tan(wind_angle) * sail_stretching / 4
    
    opt_aoa = sin(wind_angle) / (cos(wind_angle) + .4 * cos(wind_angle)**2) * sail_stretching / 4
    
    if abs(opt_aoa) > stall_deg/180.*pi:
        opt_aoa = sign(wind_angle) * stall_deg/180.*pi
    # heading controllability at high wind speeds:
    if wind_speed > limit_wind_speed:
        fact = (limit_wind_speed / wind_speed)**2
        opt_aoa *= fact
    
    return abs(clip(wind_angle - opt_aoa, -pi/2, pi/2))
    
