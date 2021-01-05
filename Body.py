import math
import matplotlib.pyplot as plt
import numpy as np 
from numpy.linalg import norm
from matplotlib.animation import FuncAnimation

class Body(object):

    # Create an object for each planet with the following parameters: 
    # color, mass, initial distance to Sun and initial acceleration
    def __init__(self, colour, mass, radius, vel_x, vel_y, acceleration):
        self.colour = colour
        self.mass = mass
        self.position = np.array( [radius, 0.0] )
        self.velocity = np.array( [vel_x, vel_y] )
        self.current_acceleration = np.array([ acceleration, 0.0 ])
        self.previous_acceleration = np.array([ acceleration, 0.0 ])
        self.timestep = 5 * 10**5
        self.half_period = []

    # Update the position of the body due to the attractive force towards Body B using Beeman Algorithm
    def update_position(self):
        self.position += self.velocity * self.timestep + (1/6) * (4 * self.current_acceleration - self.previous_acceleration) * ( self.timestep )**2

    # Update the velocity of the body due to the attractive force towards Body B using Beeman Algorithm
    def update_velocity(self, new_acc):
        self.velocity += (1/6) * (2 * new_acc + 5 * self.current_acceleration - self.previous_acceleration) * self.timestep

    # Calculate the kinetic energy for each planet
    def calculate_KE(self):
        KE = 1/2 * self.mass * ( norm(self.velocity) )**2
        return KE
    