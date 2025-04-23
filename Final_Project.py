
#All library imports go here.  
import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from NBodySimulatorClass import NBodySimulator
from NBodySimulatorClass import Get_User_input
from AnimationClass import NBodyAnimation
from IPython.display import HTML
#ran into size limits, this should fix that. I have changed how many points are simulated so I may not need this change anymore, but going to leave it  
import matplotlib as mpl
mpl.rcParams['animation.embed_limit'] = 200  # Set limit to 200 MB

 
def time_function(d):
    '''
    Okay, should take days and spit back my t_span, t_step, and t_eval
    '''
    t_span = 86400*d  # seconds
    t_step = int(np.ceil((d/100)*864)) # This should make it so that at most I'll get only 10000 points to evalulate.  Add a 0 to 864 to reduce no. of points and speed simulation.  
    t_eval = np.arange(0, t_span, t_step)
    return t_span, t_step, t_eval


 
def User_Inputs():
    '''
    Requires no initial input when called.  
    Will prompt user for their own inputs, or allow them to select from several premade simulations.
    The random simulation only works sometimes.  
    '''
    Si = 0
    masses = 0
    positions = np.array([0,0])
    velocities = np.array([0,0])
    days = 0 
    G = 6.67e-11
    Y = str(input('would you like to input your own data? Y/N'))
    if Y == 'Y' or Y == 'y':
        masses, positions, velocities, days, integrator, G = Get_User_input()
    else:
        Si = int(input('Which simulation would you like to see?\n'
            '1: A simulation of our galaxy?\n'
            '2: Our galaxy if a new star appeared at the edge of it?\n'
            '3: Inner solar system only? \n'
            '4: Something random? (due to random nature it may fail.  Use at own risk)\n'))
        integrator = str(input('please select your integrator: R for RK45, D for DOP853, A for Radau, O for ode'))
    if Si == 0:
        sim = NBodySimulator(masses, positions, velocities, Gravity = G)
        t_span, t_step, t_eval = time_function(days)

    if Si == 1:
        # All the planets in the solar system.  No moons, or other debri.  
        days = 10000
        masses = [1.99e30, 3.30e23, 4.87e24, 5.97e24, 6.42e23, 1.90e27, 5.68e26, 8.68e25, 1.02e26]
        positions = np.array([[0,0], [5.79e10,0], [1.08e11,0], [0,1.50e11], [0,2.28e11], [7.78e11,0], [1.43e12,0], [2.87e12,0], [4.50e12,0]])
        velocities = np.array([[0,0], [0,4.79e4], [0,3.5e4], [2.98e4,0], [2.41e4,0], [0, 1.31e4], [0,9.7e3], [0,6.8e3], [0,5.43e3]])
        t_span, t_step, t_eval = time_function(days)
        sim = NBodySimulator(masses, positions, velocities, Gravity = G)
        
    if Si == 2:
        #I like this one.
        days = 6000
        masses = [1.99e30, 3.30e23, 4.87e24, 5.97e24, 6.42e23, 1.90e27, 5.68e26, 8.68e25, 1.02e26,1.99e32]
        positions = np.array([[0,0], [5.79e10,0], [1.08e11,0], [0,1.50e11], [0,2.28e11], [7.78e11,0], [1.43e12,0], [2.87e12,0], [4.50e12,0],[5e12,8e5]])
        velocities = np.array([[0,0], [0,4.79e4], [0,3.5e4], [2.98e4,0], [2.41e4,0], [0, 1.31e4], [0,9.7e3], [0,6.8e3], [0,5.43e3], [-2e3,0]])
        t_span, t_step, t_eval = time_function(days)
        sim = NBodySimulator(masses, positions, velocities, Gravity = G)
    
    if Si == 3:
        #Only doing 4 planets, and the sun, gives us a more zoomed in picture.  
        days = 1000
        masses = [1.99e30, 3.30e23, 4.87e24, 5.97e24, 6.42e23]
        positions = np.array([[0,0], [5.79e10,0], [1.08e11,0], [0,1.50e11], [0,2.28e11]])
        velocities = np.array([[0,0], [0,4.79e4], [0,3.5e4], [2.98e4,0], [2.41e4,0]])
        t_span, t_step, t_eval = time_function(days)
        sim = NBodySimulator(masses, positions, velocities, Gravity = G)
        
        
    if Si == 4:
        #This should generate a random simulation and simulate it for a much shorter time.  
        #This also fails somewhat consistently, not going to remove it and don't have the time to trouble shoot.
        #often times only two bodies do anything interesting, and that's only really visible in the state space plots.  
        days =.05
        masses = [5.0, 5.0, 5.0, 5.0]
        positions = np.random.randint(-100, 101, size=(4, 2))   
        velocities = np.random.randint(-100, 101, size=(4, 2))  
        t_span, t_step, t_eval = time_function(days)
        sim = NBodySimulator(masses, positions, velocities, Gravity=9000)

    #Uses selected integrator on our sim class.    
    if integrator == 'R' or integrator == 'r':
        sol = sim.ivp_solve((0, t_span), t_eval, 'RK45' )
    if integrator == 'D' or integrator == 'd':
        sol = sim.ivp_solve((0, t_span), t_eval, 'DOP853')
    if integrator == 'A' or integrator == 'a':
        sol = sim.ivp_solve((0, t_span), t_eval, 'Radau')
    #I'm having lots of problems with ode integrator right now.  It doesn't handle the animations well, and has some annoying artifacting.
    #also the last integrator I decided to add.  I'll leave it in since it works for most of the other bits. 
    if integrator == 'O' or integrator == 'o':
        sol = sim.ode_solve(t_eval)

    #This is here to make the animation not take forever.  Might not be needed anymore since the changes to the time function.   
    n = len(masses)
    frames = sol.y[:2 * n].reshape((n, 2, -1))     # shape: (n, 2, time)
    no_of_frames = len(frames[0, 0])
    frame_stride = 1
    while  no_of_frames > 2000:
        no_of_frames = no_of_frames / 10
        frame_stride = frame_stride*10
    #was included to make something work.  Forget exactly what.  
    if frame_stride > sim.solution.t.size:
        frame_stride = max(1, sim.solution.t.size // 10)
    
    anim = NBodyAnimation(sim,frame_stride=frame_stride)
    sim.plot("n-body simulation")
    #for some reason it won't display the plot and animation at the same time but passing the anim class out works.  
    return anim
 


anim = User_Inputs()
anim.animate()

#This allows one to save an animation.  I wanted to include it with the call to make the animation, but that caused issues. 
S = str(input('would you like to save your animation? Y/N'))
if S == 'y' or S == 'Y':
    SN = str(input('what would you like it be saved as? Do not include file extension'))
    anim.save_animation(f"{SN}.mp4", fps=60)



days = 100
masses = [3.30e23, 4.87e24, 5.97e24, 6.42e23]
positions = np.array([[5.79e10,0], [1.08e11,0], [0,1.50e11], [0,2.28e11]])
velocities = np.array([[0,4.79e4], [0,3.5e4], [2.98e4,0], [2.41e4,0]])
t_span, t_step, t_eval = time_function(days)
sim = NBodySimulator(masses, positions, velocities, Gravity = 6.67e-5)
sol = sim.ivp_solve((0, t_span), t_eval, 'RK45' )
n = len(masses)
frames = sol.y[:2 * n].reshape((n, 2, -1))     # shape: (n, 2, time)
no_of_frames = len(frames[0, 0])
frame_stride = 1
while  no_of_frames > 2000:
    no_of_frames = no_of_frames / 10
    frame_stride = frame_stride*10

anim = NBodyAnimation(sim,frame_stride=frame_stride)
sim.plot("n-body simulation")
anim.animate()

S = str(input('would you like to save your animation? Y/N'))
if S == 'y' or S == 'Y':
    SN = str(input('what would you like it be saved as? Do not include file extension'))
    anim.save_animation(f"{SN}.mp4", fps=60)





