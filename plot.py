#!/usr/bin/env python3

from sys import stdin, argv
import matplotlib.pyplot as plt
from math import cos, sin


def plot(angle_chain, fig_name):
    plt.figure()
    plt.axis('equal')
    angle_chain = angle_chain.strip()
    chain = [float(x) for x in angle_chain.split(',')]
    x = [0]
    y = [0]
    cumulative = 0
    for i in range(0, len(chain)):
        if i == len(chain) - 1:
            chart_title = "Energy: " + str(chain[i])
            print(chart_title)
            continue
        cumulative += chain[i]
        x.append(x[-1] + cos(cumulative))
        y.append(y[-1] + sin(cumulative))
    plt.plot(x, y, 'o-')
    plt.title(chart_title)
    plt.savefig(fig_name)
    plt.close()
    # print(x)
    # print(y)


if __name__ == "__main__":
    count = 0
    for line in stdin:
        print(line)
        if argv[1] != '':
            figname = argv[1] + '/' + "Figure " + str(count)
        else:
            figname = "Figure " + str(count)
        plot(line, figname)
        count += 1
