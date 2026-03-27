import numpy as np
import matplotlib.pyplot as plt

GM = 1.0


def derivatives(state):
    x, y, vx, vy = state
    r = np.sqrt(x**2 + y**2)

    ax = -GM * x / r**3
    ay = -GM * y / r**3

    return np.array([vx, vy, ax, ay])


def euler_step(state, dt):
    return state + dt * derivatives(state)


def rk4_step(state, dt):
    k1 = derivatives(state)
    k2 = derivatives(state + 0.5 * dt * k1)
    k3 = derivatives(state + 0.5 * dt * k2)
    k4 = derivatives(state + dt * k3)

    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def total_energy(state):
    x, y, vx, vy = state
    r = np.sqrt(x**2 + y**2)
    v2 = vx**2 + vy**2
    return 0.5 * v2 - GM / r


def simulate(method, state0, dt, steps):
    states = np.zeros((steps + 1, 4))
    energies = np.zeros(steps + 1)

    states[0] = state0
    energies[0] = total_energy(state0)

    state = state0.copy()

    for i in range(1, steps + 1):
        state = method(state, dt)
        states[i] = state
        energies[i] = total_energy(state)

    return states, energies


state0 = np.array([1.0, 0.0, 0.0, 1.0])

dt = 0.01
t_max = 20
steps = int(t_max / dt)
time = np.linspace(0, t_max, steps + 1)


states_euler, energy_euler = simulate(euler_step, state0, dt, steps)
states_rk4, energy_rk4 = simulate(rk4_step, state0, dt, steps)

x_euler, y_euler = states_euler[:, 0], states_euler[:, 1]
x_rk4, y_rk4 = states_rk4[:, 0], states_rk4[:, 1]

plt.figure(figsize=(7, 7))
plt.plot(x_euler, y_euler, label="Euler")
plt.plot(x_rk4, y_rk4, label="RK4")
plt.scatter(0, 0, marker="*", s=200, label="Star")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Orbit Simulation: Euler vs RK4")
plt.axis("equal")
plt.grid(True)
plt.legend()

plt.savefig("orbit_comparison.png", dpi=300)  
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(time, energy_euler, label="Euler")
plt.plot(time, energy_rk4, label="RK4")
plt.xlabel("Time")
plt.ylabel("Total Energy per Unit Mass")
plt.title("Energy Comparison")
plt.grid(True)
plt.legend()

plt.savefig("energy_comparison.png", dpi=300)  # save image
plt.show()