import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from IPython.display import HTML
import multiprocessing
from NBodySimulatorClass2 import NBodySimulator

class NBodyAnimation:
    def __init__(self, simulator, interval=50, frame_stride=2000, trail_length=3000):
        self.sim = simulator
        self.interval = interval
        self.frame_stride = frame_stride
        self.trail_length = trail_length  # Max number of trail segments

    # Apply frame stride
        if frame_stride > 1:
            self.sim.solution.t = self.sim.solution.t[::frame_stride]
            self.sim.solution.y = self.sim.solution.y[:, ::frame_stride]
        self.positions_over_time = self.sim.solution.y[:2 * self.sim.n].reshape((self.sim.n, 2, -1))
        self.frames = self.positions_over_time.shape[2]

    # Set up figure
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.ax.set_title(f"N-Body Simulation with {self.sim.n} Bodies")

        # Autoscale axes
        all_positions = self.positions_over_time.reshape(self.sim.n * 2, -1)
        x_min, x_max = np.min(all_positions[0::2]), np.max(all_positions[0::2])
        y_min, y_max = np.min(all_positions[1::2]), np.max(all_positions[1::2])
        #This should help keep things legible with the legends inclusion.  Probably.  
        if x_min < y_min:
            y_min = x_min
        if x_max < y_max:
            x_max = y_max 
        if y_min < x_min:
            x_min = y_min
        if y_max < x_max:
            y_max = x_max
           
        padding_x = (x_max - x_min) * 0.1 or 1e7
        padding_y = (y_max - y_min) * 0.1 or 1e7
        self.ax.set_xlim(x_min - padding_x, x_max + padding_x)
        self.ax.set_ylim(y_min - padding_y, y_max + padding_y)

        # Initialize artists with labels including mass
        self.dots = []
        for i in range(self.sim.n):
            mass = self.sim.masses[i]
            dot, = self.ax.plot([], [], 'o', label=f'Body {i + 1} (m={mass:.2e} kg)')
            self.dots.append(dot)

        self.trail_collections = [LineCollection([], linewidths=1.5, cmap='viridis') for _ in range(self.sim.n)]
        for trail in self.trail_collections:
            self.ax.add_collection(trail)

        self.ax.legend(loc='upper left',fontsize='x-small') 

        self.xdata = [[] for _ in range(self.sim.n)]
        self.ydata = [[] for _ in range(self.sim.n)]

    def init_func(self):
        for dot, trail in zip(self.dots, self.trail_collections):
            dot.set_data([], [])
            trail.set_segments([])
        return self.dots + self.trail_collections

    def update_func(self, frame):
        for i in range(self.sim.n):
            x = self.positions_over_time[i, 0, frame]
            y = self.positions_over_time[i, 1, frame]
            if frame == 0:
                self.xdata[i] = [x]
                self.ydata[i] = [y]
            else:
                self.xdata[i].append(x)
                self.ydata[i].append(y)

            if len(self.xdata[i]) > self.trail_length:
                self.xdata[i] = self.xdata[i][-self.trail_length:]
                self.ydata[i] = self.ydata[i][-self.trail_length:]

            self.dots[i].set_data([x], [y])

            # Build segments for trail fading
            if len(self.xdata[i]) >= 2:
                points = np.array([self.xdata[i], self.ydata[i]]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                alphas = np.linspace(0.0, 1.0, len(segments))  # Fading trail
                self.trail_collections[i].set_segments(segments)
                self.trail_collections[i].set_array(alphas)

        return self.dots + self.trail_collections

    def animate(self):
        self.ani = animation.FuncAnimation(self.fig, self.update_func, frames=self.frames, init_func=self.init_func, interval=self.interval, blit=False)
        return HTML(self.ani.to_jshtml())

    def save_animation(self, filename='nbody_simulation.mp4', fps=30):
        """Save the animation as an MP4 file."""
        self.ani.save(filename, fps=fps, writer="ffmpeg", dpi=300)
