import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import time
from scipy import signal

# import csv   # for debug only
# from itertools import zip_longest # for debug only

start_time = time.process_time()  # get cpu time used


def init_arrays():
    """
    Creates array for displacements (each timestep has N+2 disp values,
    inc n=0 and n=N+1 which are zero/fixed ends), filled with zeros
    n=0 is fixed, n=N+1 is fixed, n=1 to N are moving
    :return: displacement, velocity and acceleration arrays (initial zeros)

    """
    x_arr = np.zeros((time_steps + 1, num_mass+2))
    # velocities
    vel_arr = np.zeros((time_steps + 1, num_mass+2))
    # accelerations
    acc_arr = np.zeros((time_steps + 1, num_mass+2))
    return x_arr, vel_arr, acc_arr


def init_energy_arrays():
    """
    creates the energy arrays for the first 5 modes
    :return: ak_array & modal energy array (zeros)

    """
    # ak_array is Ak in Dauxois et al, ek_array is total energy for that mode
    ak_arr = np.zeros((time_steps + 1, 6))
    ek_arr = np.zeros((time_steps + 1, 6))
    return ak_arr, ek_arr


def modal_freq():
    """
    Gets first 6 modal frequencies
    :return: modal frequencies

    """
    # k is mode (1-6) index (0-5)
    # (for N=2 should get modes sqrt(1) and sqrt(3) so equation below is correct)
    mod_freq6 = np.zeros(6)
    for kk in range(0, 6):
        mod_freq6[kk] = 2*math.sin((math.pi*(kk+1))/(2*num_mass + 2))
    return mod_freq6


def initial_disp():
    """
    :return: initial displacement array

    """
    # Initial displacement (time = 0)
    n_array = np.arange(0, num_mass + 2, 1)
    # This creates an array, start = 0, end (not inc in array) N+2, increment 1
    # n_array = [0,1,2,3.......N+1]
    if ini_choice == 0:  # half sine
        x_array[0] = amp * np.sin(np.pi * n_array / (num_mass + 1))
    if ini_choice == 1:  # full sine
        x_array[0] = amp * np.sin(2.0 * np.pi * n_array / (num_mass + 1))
    if ini_choice == 2:  # pulse
        x_array[0] = amp * signal.gausspulse(n_array, 0.1, 1)
    if ini_choice == 3:  # triangle
        for i in range(1, int((num_mass/2) + 1)):
            x_array[0, i] = amp * (2 / num_mass) * i
        x_array[0, int(num_mass/2+1)] = amp
        for i in range(int((num_mass/2) + 2), num_mass + 1):
            x_array[0, i] = amp - x_array[0, int(i-(num_mass/2)-1)]
    x_array[0, 0] = 0   # fixed ends
    x_array[0, num_mass + 1] = 0  # fixed ends
    return x_array[0]


def plot_ini():
    """
    plot initial disp
    :return: plot of initial displacement

    """
    x_axis = np.linspace(0, num_mass + 1, num=num_mass + 2)
    plt.title('initial displacement of masses')
    min_y = np.min(x_array[0])
    max_y = np.max(x_array[0])
    plt.axis([0, num_mass + 1, min_y * 1.1, max_y * 1.1])
    plt.plot(x_axis, x_array[0], 'go--')
    plt.xlabel("mass number")
    plt.ylabel("displacement")
    plt.show()


def energy():
    """
    Calculates Ak and Ek (total energy) for each mode,
    ak_array is Ak in Dauxois et al, ek_array is total energy for that mode
    :return: ak_array & modal energy array

    """
    n_array = np.arange(0, num_mass + 2, 1)
    # This creates an array, start = 0, end (not inc in array) N+2, increment 1
    # n_array = [0,1,2,3.......N+1]
    n_array[0] = 0  # fixed ends
    n_array[num_mass + 1] = 0  # fixed ends
    ak_array[j, k] = np.sqrt(2.0 / (num_mass + 1)) * np.dot(x_array[j],
                                                            np.sin((n_array * (k + 1) * math.pi / (num_mass + 1))))
    # initial profile, time = 0, j = 0
    if j == 0:
        ek_array[j, k] = 0.5 * (modal_freq6[k] ** 2 * ak_array[j, k] ** 2)
    # time > 0
    else:
        ek_array[j, k] = 0.5 * (((ak_array[j, k] - ak_array[j-1, k]) / del_t) ** 2
                                + modal_freq6[k] ** 2 * ak_array[j, k] ** 2)
    return ak_array[j, k], ek_array[j, k]


def accn(u):
    """
    gets acceleration for mass n at timestep j
    :param u: x_array = displacements
    :return: acceleration array

    """
    # Dauxois et al EQU 1
    acc_array[j, n] = (u[j, n+1] - 2 * u[j, n] + u[j, n-1]) + \
        alpha * ((u[j, n+1] - u[j, n]) ** 2 - (u[j, n] - u[j, n-1]) ** 2) + \
        beta * ((u[j, n+1] - u[j, n]) ** 3 - (u[j, n] - u[j, n-1]) ** 3)
    return acc_array[j, n]


def plot_energy():
    """
    Plot modal energies, First plot against time, second against no of mode1 cycles
    :return: plot of modal energies

    """
    x_axis = np.linspace(0, tot_time, num=time_steps + 1)
    plt.title('Mode Energy vs Time')
    min_y = 0.0
    max_y = np.max(ek_array) * 1.1
    plt.axis([0, tot_time, min_y, max_y])
    plt.plot(x_axis, ek_array[:, 0], 'g', label='M1')
    plt.plot(x_axis, ek_array[:, 1], 'b', label='M2')
    plt.plot(x_axis, ek_array[:, 2], 'r', label='M3')
    plt.plot(x_axis, ek_array[:, 3], 'k', label='M4')
    plt.plot(x_axis, ek_array[:, 4], 'm', label='M5')
    plt.plot(x_axis, ek_array[:, 5], 'c', label='M6')
    plt.legend(loc="upper right")
    plt.xlabel("Time (normalised)")
    plt.ylabel("Mode Energy (normalised)")
    text1 = "alpha {}, beta {}, time step {}".format(alpha, beta, del_t)
    plt.text(tot_time/10, max_y * 0.95, text1, fontsize=9)  # 0,3.2 is x and y location on axes
    plt.show()
    #
    # Second plot against Omega1.time/2.pi (as in Dauxois) = no of mode1 cycles
    max_omega_t = modal_freq6[0]*tot_time/2/math.pi
    x_axis = np.linspace(0, max_omega_t, num=time_steps + 1)
    plt.title('Mode Energy vs mode1 cycles')
    min_y = 0.0
    max_y = np.max(ek_array) * 1.1
    plt.axis([0, max_omega_t, min_y, max_y])
    plt.plot(x_axis, ek_array[:, 0], 'g', label='M1')
    plt.plot(x_axis, ek_array[:, 1], 'b', label='M2')
    plt.plot(x_axis, ek_array[:, 2], 'r', label='M3')
    plt.plot(x_axis, ek_array[:, 3], 'k', label='M4')
    plt.plot(x_axis, ek_array[:, 4], 'm', label='M5')
    plt.plot(x_axis, ek_array[:, 5], 'c', label='M6')
    plt.legend(loc="upper right")
    plt.xlabel("mode 1 cycles")
    plt.ylabel("Mode Energy (normalised)")
    text1 = "alpha {}, beta {}, time step {}".format(alpha, beta, del_t)
    plt.text(max_omega_t/10, max_y * 0.95, text1, fontsize=9)  # 0,3.2 is x and y location on axes
    plt.show()
    #


def anim_disp():
    """
    animate disps
    :return: animation of displacements (gif file)

    """
    j_start = int(an_start / del_t)
    j_end = int(an_end / del_t)
    x_axis = np.arange(0, num_mass + 2, 1)
    mass_image = []
    max_amp = np.max(x_array)
    min_amp = np.min(x_array)
    fig_anim = plt.figure(dpi=70)
    for jj in x_array[j_start:j_end, :]:
        mass_image.append(plt.plot(x_axis, jj, 'ro'))
    anim1 = animation.ArtistAnimation(fig_anim, mass_image, interval=10, repeat=False, blit=True)
    plt.axis([0, num_mass + 2, min_amp * 1.1, max_amp * 1.1])
    plt.xlabel("Mass Number")
    plt.ylabel("Displacement")
    text1 = "alpha {}, beta {}, time step {}, start time {}, end time {} ".format(alpha, beta, del_t, an_start, an_end)
    plt.text(1, max_amp * 0.5, text1, fontsize=9)  # x and y location on axes
    # Save as Gif-file:
    # writergif = animation.PillowWriter(fps=30)
    writergif = animation.PillowWriter(fps=20)
    anim1.save('anim_disp.gif', writer=writergif)
    plt.close()
    #


def anim_vel():
    """
    animate velocities
    :return: animation of velocities (gif file)

    """
    j_start = int(an_start / del_t)
    j_end = int(an_end / del_t)
    x_axis = np.arange(0, num_mass + 2, 1)
    mass_image = []
    max_amp = np.max(vel_array)
    min_amp = np.min(vel_array)
    fig_anim = plt.figure(dpi=70)
    for jj in vel_array[j_start:j_end, :]:
        mass_image.append(plt.plot(x_axis, jj, 'ro'))
    anim1 = animation.ArtistAnimation(fig_anim, mass_image, interval=10, repeat=False, blit=True)
    plt.axis([0, num_mass + 2, min_amp * 1.1, max_amp * 1.1])
    plt.xlabel("Mass Number")
    plt.ylabel("Velocity")
    text1 = "alpha {}, beta {}, time step {}, start time {}, end time {} ".format(alpha, beta, del_t, an_start, an_end)
    plt.text(1, max_amp * 0.5, text1, fontsize=9)  # x and y location on axes
    # Save as Gif-file:
    # writergif = animation.PillowWriter(fps=30)
    writergif = animation.PillowWriter(fps=20)
    anim1.save('anim_vel.gif', writer=writergif)
    plt.close()
    #


def plot_snap_disp():
    """
    plot of displacements at chosen time
    :return: plot of displacements at chosen time

    """
    jj = int(snap_time / del_t)
    # returns num_mass+2 evenly spaced numbers over range 0 to num_mass+1
    x_axis = np.linspace(0, num_mass + 1, num=num_mass + 2)
    text1 = "displacement field at time {}".format(snap_time)
    plt.title(text1)
    min_y = np.min(x_array[jj, :])
    max_y = np.max(x_array[jj, :])
    plt.axis([0, num_mass + 1, min_y * 1.1, max_y * 1.1])
    plt.plot(x_axis, x_array[jj], 'go-')
    plt.xlabel("mass number")
    plt.ylabel("displacement")
    text2 = "alpha {}, beta {}, time step {}".format(alpha, beta, del_t)
    plt.text(num_mass / 4, max_y * 0.8, text2, fontsize=9)  # x and y location on axes
    plt.show()
    #


def plot_snap_vel():
    """
    plot of velocities at chosen time
    :return: plot of velocities at chosen time

    """
    jj = int(snap_time / del_t)
    # returns num_mass+2 evenly spaced numbers over range 0 to num_mass+1
    x_axis = np.linspace(0, num_mass + 1, num=num_mass + 2)
    text1 = "velocity field at time {}".format(snap_time)
    plt.title(text1)
    min_y = np.min(vel_array[jj, :])
    max_y = np.max(vel_array[jj, :])
    plt.axis([0, num_mass + 1, min_y * 1.1, max_y * 1.1])
    plt.plot(x_axis, vel_array[jj], 'go-')
    plt.xlabel("mass number")
    plt.ylabel("velocity")
    text2 = "alpha {}, beta {}, time step {}".format(alpha, beta, del_t)
    plt.text(num_mass / 4, max_y * 0.8, text2, fontsize=9)  # x and y location on axes
    plt.show()
    #


def plot_disp_cont():
    """
    plot contour of disp vs time and mass no
    :return: contour plot of disp vs time and mass no

    """
    j_start = int(min_time / del_t)
    j_end = int(max_time / del_t)
    steps = j_end - j_start
    cont_array = x_array[j_start:j_end, :]
    x_axis = np.arange(0, num_mass + 2, 1)
    y_axis = np.linspace(min_time, max_time, num=steps)
    max_amp = np.max(cont_array)
    min_amp = np.min(cont_array)
    levels = [min_amp, 0.8*min_amp, 0.6*min_amp, 0.4*min_amp, 0.2*min_amp, 0.0,
              0.2*max_amp, 0.4*max_amp, 0.6*max_amp, 0.8*max_amp, max_amp]
    plt.contourf(x_axis, y_axis, cont_array, levels, cmap='PiYG')
    plt.colorbar()
    plt.title('displacement contours vs mass number & time')
    plt.xlabel("Mass Number")
    plt.ylabel("Time")
    text1 = "alpha {}, beta {}, time step {}".format(alpha, beta, del_t)
    plt.text(num_mass / 4, max_time * 0.9, text1, fontsize=9)  # x and y location on axes
    plt.show()
    #


def plot_vel_cont():
    """
    plot contour of velocity vs time and mass no
    :return: contour plot of velocity vs time and mass no

    """
    # plot contour of velocity vs time and mass no
    j_start = int(min_time / del_t)
    j_end = int(max_time / del_t)
    steps = j_end - j_start
    cont_array = vel_array[j_start:j_end, :]
    x_axis = np.arange(0, num_mass + 2, 1)
    y_axis = np.linspace(min_time, max_time, num=steps)
    max_amp = np.max(cont_array)
    min_amp = np.min(cont_array)
    levels = [min_amp, 0.8*min_amp, 0.6*min_amp, 0.4*min_amp, 0.2*min_amp, 0.0,
              0.2*max_amp, 0.4*max_amp, 0.6*max_amp, 0.8*max_amp, max_amp]
    plt.contourf(x_axis, y_axis, cont_array, levels, cmap='PiYG')
    plt.colorbar()
    plt.title('velocity contours vs mass number & time')
    plt.xlabel("Mass Number")
    plt.ylabel("Time")
    text1 = "alpha {}, beta {}, time step {}".format(alpha, beta, del_t)
    plt.text(num_mass / 4, max_time * 0.9, text1, fontsize=9)  # x and y location on axes
    plt.show()
    #

# **************************END OF FUNCTIONS******************************
# **************************END OF FUNCTIONS******************************
# **************************END OF FUNCTIONS******************************
#
# INPUTS;
#


num_mass = 32  # number of moving masses (excluding fixed ends)
tot_time = 12000.  # 12000 #10000 Total time to analyse
del_t = 0.2  # timestep
alpha = 0.25  # coefficient for quadratic correction on Hooke's Law
beta = 0.0  # coefficient for cubic correction on Hooke's Law
#
# Initial displacement, time = 0, x_array[] is displacements
# choice is 0 = half sine wave, 1 = full sine wave, 2 = pulse, 3 = triangle
# (NOTE: initial velocity is zero in each case)
ini_choice = 0
amp = 1  # amplitude choice

# Calc total no of timesteps
time_steps = int(tot_time/del_t)
#
# Initialise displacement, velocity and acceleration arrays
x_array, vel_array, acc_array = init_arrays()
#
# Initialise energy arrays
ak_array, ek_array = init_energy_arrays()
# print(ak_array)
# print(ek_array)
#
# Get first 6 modal frequencies
modal_freq6 = modal_freq()
# print(modal_freq6)
#
# Apply initial displacement
x_array[0] = initial_disp()
#
# Plot initial displacement
plot_ini()
#
# Mode and total energy for initial displacement
# j is timestep number [0,1,2......time_steps]
# k is mode number
j = 0  # timestep number = 0 for initial displacement
for k in range(0, 6):
    ak_array[j, k], ek_array[j, k] = energy()
# debug...
# print(ak_array[0])
# print(ek_array[0])

#
# Time Loop (timestep no is j)
# *********
# "for in range loops" inc start value but exclude the stop value (so +1 to stop value)
for j in range(1, time_steps+1):

    # VELOCITY VERLET INTEGRATION SCHEME
    #  https://en.wikipedia.org/wiki/Verlet_integration
    # Dont use half-step velocity as acceleration only depends on displacement
    # not velocity
    # t = j-1, t+del_t = j
    # calc ALL n new disps x(j) = x(j-1) + v(j-1)*del_t + 0.5*a(j-1)*(del_t)**2
    # get a(j) from new disp array = Dauxois et al Equ 1 using x(j)
    # Calc v(j) = v(j-1) + 0.5 * [a(j-1) + a(j)] * del_t
    # *****************************************************
    # Calc ALL [j,:] does all masses N+2 (inc 2 ends) for timestep j
    x_array[j, :] = x_array[j-1, :] + vel_array[j-1, :] * del_t \
                   + 0.5 * acc_array[j-1, :] * (del_t ** 2)
    # Moving mass loop (mass is n), n=0 & n=N+1 are fixed
    for n in range(1, num_mass + 1):
        # Dauxois et al EQU 1
        acc_array[j, n] = accn(x_array)
        vel_array[j, n] = vel_array[j-1, n] + 0.5 * (acc_array[j-1, n] + acc_array[j, n]) * del_t

    # Mode and total energy for each timestep
    # j is timestep number [0,1,2......time_steps]
    # k is mode number
    for k in range(0, 6):
        ak_array[j, k], ek_array[j, k] = energy()

#
# Do plots of energy vs time
#
plot_energy()

#
# Animation of disps and vels
# ****NOTE - DOING THIS SIGNIFICANTLY INCREASES RUN TIME
# Just uncomment when want
an_start = 10000  # animation start time
an_end = 10500  # animation end time
# anim_disp()  # animate disps
# anim_vel()  # animate vels
#

#
# Plot a snap shot or velocity field at a chosen time
#
snap_time = 4929  # pick time for snap-shot
plot_snap_disp()
plot_snap_vel()

#
# Contour plot of disp & vel vs time (y) and mass no (x)
#
# time range required:
min_time = 0  # 4500 # 0
max_time = tot_time  # tot_time
plot_disp_cont()
plot_vel_cont()


# debug.......
# print(ek_array[:,0])
# print(x_array)

#
# TEMPORARY DEBUG = WRITE x_array TO CSV to check Verlet
#
# export_data = zip_longest(*x_array, fillvalue='')
# with open('x_array_debug.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
#    wr = csv.writer(myfile)
#    wr.writerow(("t0", "t1", "t2", "t3", "t4", "t5", "t6"))
#    wr.writerows(export_data)
# myfile.close()


# Get CPU time
end_time = time.process_time()
cpu_time = end_time - start_time
print('CPU time:', cpu_time, 's')
