import math
import matplotlib.pyplot as plt
import numpy as np 
from numpy.linalg import norm
from matplotlib.animation import FuncAnimation
from Body import Body

class Simulation(object):

    # Create the simulation for all of the planets using the body list, the patch list,
    # the no of iterations and the timestep
    def __init__(self, num_iterations, body_list, patches):
        self.time = 0
        self.time_flow = []
        self.kin_energy = []
        self.pot_energy = []
        self.total_energy = []
        self.num_iterations = num_iterations
        self.body_list = body_list 
        self.patches = patches
        self.gamma = 6.67 * 10**(-11) 

    # Read the input values for the colour, mass, distance and velocity of each planet wrt the Sun
    def read_values(self):
        # Introduce the number of planets
        N = int( input("The number of the planets is: ") )
        for i in range (N):
            colour =  input("The color of planet " + str(i+1) + " is: ") 
            mass =  ( float( input("The mass of planet " + str(i+1) + " in Earth masses is: ") ) ) * (6 * 10**24)
            if(i != 0):
                # Write the position of the planet
                position_x =  ( float( input("The distance of planet " + str(i+1) + " wrt the Sun in AU is: ") ) ) * (1.5 * 10**11)
                # Write the speed wrt to the circular trajectory speed
                circular_velocity = math.sqrt(self.gamma * self.body_list[0].mass / position_x )
                velocity_x =  ( float( input("The fraction of circular trajectory speed of planet on x axis " + str(i+1) + " wrt the Sun is: ") ) ) * circular_velocity
                velocity_y =  ( float( input("The fraction of circular trajectory speed of planet on y axis " + str(i+1) + " wrt the Sun is: ") ) ) * circular_velocity
                velocity = np.array( [velocity_x, velocity_y] )
                # Write the current and previous accelerations of each planet
                initial_acc =  -self.gamma * self.body_list[0].mass / (position_x)**2
            else:
                position_x = 0.0
                velocity_x = 0.0
                velocity_y = 0.0
                initial_acc = 0.0
                previous_acc = 0.0
            # Create the Body B and append it to the body list
            B = Body(colour, mass, position_x, velocity_x, velocity_y, initial_acc)
            self.body_list.append(B)

    # Calculate acceleration of Body B wrt the other planets
    def calculate_acceleration(self, B):
        acc = np.array( [0.0, 0.0] )
        for i in range (len(self.body_list)):
            if(self.body_list[i] != B):
                delta_d = self.body_list[i].position - B.position
                distance = norm (self.body_list[i].position - B.position)
                acc_i = self.gamma * self.body_list[i].mass / (distance)**2
                angle = np.array( [delta_d[0] / distance, delta_d[1] / distance] )
                acc += acc_i * angle
        return acc

    # Calculate the potential energy for the whole system
    def calculate_PE(self):
        PE = 0.0
        for i in range (len(self.body_list)):
            for j in range (i+1, len(self.body_list)):
                distance = norm (self.body_list[i].position - self.body_list[j].position)
                PE += -self.gamma * self.body_list[i].mass * self.body_list[j].mass / distance
        return PE

    # Define initiator for the animate function
    def init(self):
        return self.patches

    # Advance after a small timestep and update all the parameters of each Body B
    def step_forward(self, i):
        # Update position of each planet
        for i in range (len(self.body_list)):
            # Check if period has been made
            y_init = self.body_list[i].position[1]
            self.body_list[i].update_position()
            y_final = self.body_list[i].position[1]
            ratio = y_final / y_init
            if(ratio < 0):
                self.body_list[i].half_period.append(self.time)
        for i in range (len(self.body_list)):
            # Calculate new acceleration for each planet
            acc = self.calculate_acceleration(self.body_list[i])
            # Update the velocity of each planet
            self.body_list[i].update_velocity(acc)
            # Update the previous and current accelerations
            self.body_list[i].previous_acceleration = self.body_list[i].current_acceleration
            self.body_list[i].current_acceleration = acc
        # Update the positions of the patches
        for i in range (len(self.patches)):
            self.patches[i].center = (self.body_list[i].position[0], self.body_list[i].position[1])
        # Calculate the overall kinetic energy
        kin_energy = 0
        for i in range (len(self.body_list)):
            kin_energy += self.body_list[i].calculate_KE()
        print("The kinetic energy of the system is: " + str(kin_energy))
        self.kin_energy.append(kin_energy)
        # Calculate the potential energy of the system
        pot_energy = self.calculate_PE()
        print("The potential energy of the system is: " + str(pot_energy))
        self.pot_energy.append(pot_energy)
        # Calculate the overall energy of the system
        energy = kin_energy + pot_energy
        print("The total energy of the system is: " + str(energy))
        self.total_energy.append(energy)
        # Increase the time
        self.time += 5 * 10**5
        self.time_flow.append(self.time) 
        return self.patches

    # Run the simulation
    def run_simulation(self):
        # Create plot elements
        fig = plt.figure()
        ax = plt.axes() 
        # Create the live graph for energies
        sub_fig = fig.add_subplot(1,1,1)
        # Add all the patches
        self.patches = []   
        for i in range(len(self.body_list)):
            # create circles of radius 0.1 centred at initial position and add to list
            self.patches.append(plt.Circle((self.body_list[i].position), 10**11, color = self.body_list[i].colour, animated = True))
            # add circles to axes
            ax.add_patch(self.patches[-1])
        # set up the axes
        ax.axis('scaled')
        ax.set_xlim(-3 * 10**12, 3 * 10**12)
        ax.set_ylim(-3 * 10**12, 3 * 10**12)
        ax.set_xlabel('x coordinate')
        ax.set_ylabel('y coordinate')
        # create the animator
        anim = FuncAnimation(fig, func = self.step_forward, init_func = self.init, frames = self.num_iterations, repeat = False, interval = 10, blit = True)
        # show the plot
        plt.show()

def main():
    # Initialize the body list and the patches by reading the values from keyboard
    body_list = []
    patches = []
    solar_system = Simulation(1000, body_list, patches)
    # Run the simulation
    solar_system.read_values()
    solar_system.run_simulation()
    # Plot the kinetic, potential and total energy as function of time
    plt.plot(solar_system.time_flow, solar_system.kin_energy, label = "Kinetic energy")
    plt.plot(solar_system.time_flow, solar_system.pot_energy, label = "Potential energy")
    plt.plot(solar_system.time_flow, solar_system.total_energy, label = "Total energy")
    plt.legend()
    plt.xlabel("Time step")
    plt.ylabel("Energies")
    plt.show()
    # Find out the half period of each planet
    for i in range (1, len(solar_system.body_list)):
        if(len(solar_system.body_list[i].half_period) > 2):
            print("The period of planet " + str(i) + " is: " + str((solar_system.body_list[i].half_period[-1] - solar_system.body_list[i].half_period[-3]) / (86400 * 365) ) + " Earth years " )
        else:
            print("The orbit of planet " + str(i) + " is not stable")

main()  
