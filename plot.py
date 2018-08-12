#!/usr/bin/env python3

from sys import stdin
import matplotlib.pyplot as plt
from math import cos, sin


def plot(angle_chain, fig_name):
    plt.figure()
    plt.axis('equal')
    chain = [float(x) for x in angle_chain.split()]
    x = [0]
    y = [0]
    cumulative = 0
    for alpha in chain:
        cumulative += alpha
        x.append(x[-1] + cos(cumulative))
        y.append(y[-1] + sin(cumulative))
    plt.plot(x, y)
    plt.savefig("./documentation/figure " + fig_name + ".png")
    print(chain)
    plt.close()


if __name__ == "__main__":
    count = 0
    for line in stdin:
        plot(line, str(count))
        count += 1
