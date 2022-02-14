import random
import numpy as np
import matplotlib.pyplot as plt


def GA(mode):
    # Pick a random number between 2^100 and 0. If needed adjust to be l bits long by padding 0 on left
    x = bin(random.randint(0, 2**l))[2:].rjust(l, '0')
    fitness = []
    for m in range(n_iterations):
        # Create array indicating which indices flip
        flips = np.random.choice([0, 1], l, p=[1-p, p])
        # flip bits by using xor or remain the same when they don't need to be flipped
        x_m = ''.join(format(int(bit) ^ flip, 'b') if flip else bit
                      for bit, flip in zip(x, flips))

        # Depending on which mode, check score and replace (b) or just replace (c)
        if mode == 'b':
            score = x_m.count('1')
            if score > x.count('1'):
                x = x_m
        elif mode == 'c':
            x = x_m
        else:
            print('invalid mode')
        # Save fitness (counting the 1s present) score
        fitness.append(x.count('1'))
    return fitness


l = 100
p = 1/l
n_iterations = 1500
plt.figure(figsize=(10, 10))
for i in range(10):
    fitness_scores = GA('c')
    plt.plot(np.arange(0, n_iterations), fitness_scores)
plt.xlabel('n_iterations')
plt.ylabel('fitness')
plt.ylim([0, 100])
plt.title('Fitness of the GA over iterations')
plt.show()
