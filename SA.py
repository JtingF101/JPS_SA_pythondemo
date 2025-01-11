import math
from random import random
import matplotlib.pyplot as plt


# Function optimisation problems
def func(x, y):
    res = 4 * x ** 2 - 2.1 * x ** 4 + x ** 6 / 3 + x * y - 4 * y ** 2 + 4 * y ** 4
    return res


# x is x1 in the formula, y is x2 in the formula
class SA:
    def __init__(self, func, iter=100, T0=100, Tf=0.01, alpha=0.99):
        self.func = func
        self.iter = iter  # of iterations in the inner loop, i.e. L = 100
        self.alpha = alpha  # Cooling factor, alpha = 0.99
        self.T0 = T0  # Initial temperature T0 is 100
        self.Tf = Tf  # Temperature final Tf of 0.01
        self.T = T0  # Current temperature
        self.x = [random() * 11 - 5 for i in range(iter)]  # Randomly generate 100 values of x
        self.y = [random() * 11 - 5 for i in range(iter)]  # Randomly generate 100 values of y
        self.most_best = []
        # The function random() takes a decimal number between 0 and 1
        # If you want to take an integer between 0 and 10 (including 0 and 10) just write (int)random()*11, 11 multiplied by zero is a maximum of 10 points and a minimum of 0 points
        # The absolute values of x1 and x2 in this example do not exceed 5 (including the integers 5 and -5), and the result of (random() * 11 -5) is any value between -6 and 6 (excluding -6 and 6) (random() * 10 -5) results in any value between -5 and 5 (excluding -5 and 5), all multiply by 11 first, take a value between -6 and 6, and the process of generating a new solution filters out the ones between -5 and 5 (including the integers 5 and -5) with an if conditional statement.
        self.history = {'f': [], 'T': []}

    def generate_new(self, x, y):  # Perturbation to generate new solutions
        while True:
            x_new = x + self.T * (random() - random())
            y_new = y + self.T * (random() - random())
            if (-5 <= x_new <= 5) & (-5 <= y_new <= 5):
                break  # Repeat to obtain new solutions until the resulting new solution satisfies the constraints
        return x_new, y_new

    def Metrospolis(self, f, f_new):  # Metropolis guidelines
        if f_new <= f:
            return 1
        else:
            p = math.exp((f - f_new) / self.T)
            if random() < p:
                return 1
            else:
                return 0

    def best(self):  # Get the optimal objective function value
        f_list = []  # f_list array holds the values after each iteration
        for i in range(self.iter):
            f = self.func(self.x[i], self.y[i])
            f_list.append(f)
        f_best = min(f_list)

        idx = f_list.index(f_best)
        return f_best, idx  # f_best,idx are the optimal solution of the objective function and the subscript of the optimal solution after L iterations at this temperature, respectively

    def run(self):
        count = 0
        # Outer loop iteration with current temperature less than the threshold for termination temperature
        while self.T > self.Tf:

            # 100 iterations of the inner loop
            for i in range(self.iter):
                f = self.func(self.x[i], self.y[i])  # f is the value after one iteration
                x_new, y_new = self.generate_new(self.x[i], self.y[i])  # Generate new solutions
                f_new = self.func(x_new, y_new)  # Generate new values
                if self.Metrospolis(f, f_new):  # Determine whether to accept a new value
                    self.x[i] = x_new  # If the new value is accepted, store the x,y of the new value into the x array and y array
                    self.y[i] = y_new
            # Iterate L times to record the optimal solution at that temperature
            ft, _ = self.best()
            self.history['f'].append(ft)
            self.history['T'].append(self.T)
            # Temperature decreases by a certain percentage (cooling)
            self.T = self.T * self.alpha
            count += 1

        # To get the optimal solution
        f_best, idx = self.best()
        print("F={}, x={}, y={}".format(f_best, self.x[idx], self.y[idx]))

        # print(f"F={f_best}, x={self.x[idx]}, y={self.y[idx]}")


sa = SA(func)
sa.run()

plt.plot(sa.history['T'], sa.history['f'])
plt.title('SA')
plt.xlabel('T')
plt.ylabel('f')
plt.gca().invert_xaxis()
plt.show()
