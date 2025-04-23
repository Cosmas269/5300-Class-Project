import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
#G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
import numpy as np
from scipy import integrate, optimize
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def Get_User_input(Mn=0,Dn=0,Vn=0,N=0):
    """
    Should prompt the user for three different sets of variables
    M:  Mass
    D:  Distances of our bodies from an origin in meters, and in (x,y) coordinates
    V:  Velocities of our bodies in m/s and in (x,y) coordinates. 
    """
    if N==0:
        G = 6.67430e-11
        N = int(input('please input the number of bodies that will be simulated '))
        D = float(input('how many days would you like this simulation to simulate?  Fractional Days are fine. '))
        I = str(input('please select your integrator: R for RK45, D for DOP853, A for Radau, O for ode'))
        GC = str(input('Would you like to change the strength of gravity Y/N'))
        if GC == 'Y' or GC == 'y':
            G = int(input('please input your new value of Gravity')) 
        Ma = np.zeros(N)
        Da = np.zeros((N,2))
        Va = np.zeros((N,2))
    while Mn<N:
        M = float(input(f'please input the mass of body {Mn+1} ')) 
        Ma[Mn] = M
        Mn += 1
    while Dn < N:
        input_str = input(f"Please input the x,y coordinate of body {Dn + 1}: ")
        Dx, Dy = map(float, input_str.split(','))
        Da[Dn] = [Dx, Dy]
        Dn +=1
    while Vn< N:
        input_str = (input(f'please input the x,y velocity of body {Vn+1} '))
        Vx, Vy = map(float, input_str.split(','))
        Va[Vn] = [Vx, Vy]
        Vn +=1
    
    return Ma, Da, Va, D, I, G

class NBodySimulator:
    def __init__(self, masses, positions, velocities, Gravity=6.67430e-11):
        """
        masses:    array of shape (n)
        positions: array of shape (n, 2)
        velocities: array of shape (n, 2)
        """
        self.n = len(masses)
        self.masses = np.array(masses, dtype=np.float64)
        self.positions = np.array(positions, dtype=np.float64)
        self.velocities = np.array(velocities, dtype=np.float64)
        self.G = Gravity
        self.solution = None

    def _derivatives(self, t, state):
        """
        Compute derivatives: [dx/dt, dy/dt, dvx/dt, dvy/dt] for all bodies.
        This is where the magic happens.  This should calculate the derivatives for all the bodies involved.  
        I think this is right, but when dealing with sums its hard for me to be certain.  
        """
        n = self.n
        positions = state[:2 * n].reshape((n, 2))
        velocities = state[2 * n:].reshape((n, 2))
        accelerations = np.zeros((n, 2), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                diff = positions[j] - positions[i]
                distance_squared = np.dot(diff, diff) + 1e-10
                distance_cubed = distance_squared ** 1.5
                accelerations[i] += self.G *self.masses[j] * diff / distance_cubed

        return np.hstack([velocities.flatten(), accelerations.flatten()])

    def ivp_solve(self, t_span, t_eval=None, method='RK45'):
        '''
        Parameters:
            t_span: (t0, tf) in seconds
            t_eval: array of time points to evaluate the solution at
            method: string, integration method (e.g. 'RK45', 'DOP853', 'Radau', I suspose there are other possibilities here, but I'm not going to ask for them)
        '''
        initial_state = np.hstack([self.positions.flatten(), self.velocities.flatten()])
        self.solution = solve_ivp(fun=self._derivatives, t_span=t_span, y0=initial_state, t_eval=t_eval, method=method,rtol=1e-9, atol=1e-9)
        return self.solution
          # t_span,  
    def ode_solve(self, t_eval, rtol=1e-9, atol=1e-9):
        """
        I swear this thing doesn't work right, but that might be because I'm not passing it enough points. 
        Parameters:
            t_span: (t0, tf)

        """
        def derivatives_odeint(state, t):
            return self._derivatives(t, state)

        initial_state = np.hstack([self.positions.flatten(), self.velocities.flatten()])
        trajectory = odeint(func=derivatives_odeint, y0=initial_state, t=t_eval, atol=atol, rtol=rtol)

        class OdeintSolution:
            pass

        self.solution = OdeintSolution()
        self.solution.t = t_eval
        self.solution.y = trajectory.T  # Transpose for consistency with solve_ivp
        return self.solution
        
    def plot(self, title="N-Body Simulation"):
        # Sometimes the graph gets too stretched to make sense of.  This should help limit that.  
        n = self.n
        positions = self.solution.y[:2 * n].reshape((n, 2, -1))
        velocities = self.solution.y[2 * n:].reshape((n, 2, -1))
        x_min, x_max = np.min(positions[0::2]), np.max(positions[0::2])
        y_min, y_max = np.min(positions[1::2]), np.max(positions[1::2])
        if x_min <-2e12:
            x_min = -2e12
        if x_max > 2e12:
            x_max = 2e12
        if y_min < -2e12:
            y_min = -2e12
        if y_max > 2e12:
            y_max = 2e12

        plt.figure(figsize=(10, 8))
        for i in range(n):
            plt.plot(positions[i, 0], positions[i, 1], label=f'Body {i + 1}')
        plt.xlim(x_min , x_max )
        plt.ylim(y_min , y_max )
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title(title)
        plt.axis('equal')
        plt.grid()
        plt.legend()
        plt.show()
        #Should create state space plots of radial distance vs radial velocity for each mass.  Likely to cause a mess though. 
        for i in range(n):
            r = np.linalg.norm(positions[i], axis=0)
            v = velocities[i]
            vr = np.sum(positions[i] * v, axis=0) / r  # dot(r, v) / |r|

            plt.figure(figsize=(8, 6))
            plt.plot(r, vr, label=f'Body {i + 1}')
            plt.xlabel('Radial Distance r (m)')
            plt.ylabel('Radial Velocity v_r (m/s)')
            plt.title(f'Radial Distance vs Radial Velocity — Body {i + 1}')
            plt.grid(True)
            plt.legend()
            plt.show()


        
