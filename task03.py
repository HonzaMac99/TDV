import numpy as np
import matplotlib.pyplot as plt
import toolbox as tb


def find_start_end(line):
    start = np.zeros([2,])
    end = np.zeros([2,])
    j = 0
    for i in range(2):
        while line[j] == ' ':
            j += 1
        start[i] = j

        while line[j] != ' ' and line[j] != '\n':
            j += 1
        end[i] = j

    return start, end


def read_points(fname):
    file = open(fname, "r")
    points = []
    for line in file:
        start, end = find_start_end(line)
        point_x = float(line[start[0]:end[0]])
        point_y = float(line[start[1]:end[1]])
        points.append([point_x, point_y])

    return np.array(points)


data = read_points("linefit_1.txt")
print(data)


