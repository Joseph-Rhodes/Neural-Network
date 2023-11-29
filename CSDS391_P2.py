import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv


def plot_data(d, is_title=False, title=''):
    for r in d:

        if r[4] == 'setosa':
            c = 'red'
        elif r[4] == 'versicolor':
            c = 'green'
        elif r[4] == 'virginica':
            c = 'blue'
        else:
            c = 'black'

        plt.plot(float(r[2]), float(r[3]), linestyle='none', marker='o', color=c)

    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    if is_title:
        plt.title(title)
    pass


def k_means(k, d):
    averages = k * [[0.0, 0.0]]
    for index in range(k):
        while True:
            rand_point = random.randint(0, len(d))
            if not averages.__contains__([float(d[rand_point][2]), float(d[rand_point][3])]):
                break
        averages[index] = [float(d[rand_point][2]), float(d[rand_point][3])]

    objective_function = []

    iterations = 0

    while True:
        iterations += 1

        if len(d) == 150:
            plot_data(d)
            for average in averages:
                plt.plot(average[0], average[1], color='black', marker='o', linestyle='none')

            objective_function.append(obj_function(d, averages))

            plt.title('k-means clustering for ' + str(k) + ' Clusters, Iteration: ' + str(iterations))
            plt.show()

        last_averages = averages.copy()

        averages = [[0.0, 0.0] for i in range(k)]
        num_points = k * [0.0]
        for r in d:
            index = last_averages.index(closest_mean([r[2], r[3]], last_averages))

            num_points[index] += 1.0

            averages[index][0] += float(r[2])
            averages[index][1] += float(r[3])

        for index in range(k):
            averages[index][0] /= num_points[index]
            averages[index][1] /= num_points[index]

        if last_averages == averages:
            if len(d) == 150:
                plt.plot(range(len(objective_function)), objective_function, 'ko')
                plt.plot(range(len(objective_function)), objective_function, 'k')
                plt.xlabel('Iteration')
                plt.ylabel('Sum of Error Squared')
                plt.title('Objective Function for ' + str(k) + ' Clusters')
                plt.show()
                return [averages, objective_function]
            else:
                return [averages, 0]

def obj_function(d, means):
    sse = 0.0

    for r in d:
        sse += math.pow(get_distance([float(r[2]), float(r[3])], closest_mean([float(r[2]), float(r[3])], means)), 2)

    return sse
 
def get_distance(pa, pb):

    return math.sqrt(math.pow(pa[0] - pb[0], 2) + math.pow(pa[1] - pb[1], 2))

def closest_mean(r, means):
    distance = 99.9
    closest_mean = [0.0, 0.0]

    for mean in means:
        d = get_distance([float(r[0]), float(r[1])], mean)

        if d < distance:
            distance = d
            closest_mean = [float(mean[0]), float(mean[1])]

    return closest_mean

def decision_bounds(point1, point2, t):
    x_constant = (point1[0] + point2[0]) / 2.0
    y_constant = (point1[1] + point2[1]) / 2.0

    slope = abs(point2[0] - point1[0]) / abs(point2[1] - point1[1])
    return y_constant - (t - x_constant) / slope




def get_likelihood(point, cluster, clusters):
    distance_test = 1.0 / get_distance(point, cluster)

    distance_all = 0.0
    for c in clusters:
        distance_all += 1.0 / get_distance(point, c)

    return distance_test / distance_all

def plot_decision_boundaries(num_clusters, iris_data, t):
    uk = []
    d = []

    output = k_means(num_clusters, iris_data)
    uk.append(output[0])
    d.append(output[1])
    plot_data(iris_data)

    for index in range(len(uk[0])):
        p1 = [uk[0][index][0], uk[0][index][1]]
        plt.plot(p1[0], p1[1], 'ko', linestyle='none')

        for index2 in range((index + 1), len(uk[0])):
            p2 = uk[0][index2][0], uk[0][index2][1]

            x = (p1[0] + p2[0]) / 2.0
            y = (p1[1] + p2[1]) / 2.0

            m = abs(p2[0] - p1[0]) / abs(p2[1] - p1[1])
            line = y - (t - x) / m

            l1 = get_likelihood([x, y], p1, uk[0])
            l3 = 1.0 - (2.0 * l1)
            if l3 < l1:
                plt.plot(t, line, 'c:')

    names = ['Setosa', 'Versicolor', 'virginica', 'Cluster', 'Decision Boundaries']
    colors = ['r', 'g', 'b', 'k', 'c']
    hands = []

    for i in range(5):
        hands.append(mpatches.Patch(color=colors[i], label=names[i]))

    plt.legend(handles=hands, loc='upper left')

    if len(iris_data) == 150:
        plt.xlim(0.0, 7.1)
        plt.ylim(0.0, 2.6)
    else:
        plt.xlim(2.9, 7.1)
        plt.ylim(0.9, 2.6)

    plt.title('Decision Boundaries for ' + str(num_clusters) + ' Clusters')
    plt.show()

    pass


def plot_nn(d, m, b):
    ax = plt.axes(projection='3d')
    for v in d:
        x = float(v[2])
        y = float(v[3])
        if v[4] == 'versicolor':
            color = 'b'
        else:
            color = 'g'

        plt.plot(x, y, sigmoid(m * x - y + b), 'o', color=color)

    plt.show()

    pass
        
def plot_nn_decision_bounds(d, m, b, point):
    time = np.linspace(3.0, 7.0, 200)

    plot_data(d)
    plt.plot(time, m * time + b, 'c')
    plt.ylim(0.9, 2.6)
    plt.title('Decision Boundary for Non-Linearity')
    plt.show()

    # z = mx - y + b
    return 1.0 - sigmoid(m * point[0] - point[1] + b)

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def simple_classifier(d, m, b, k):
    v = d[k]
    print('Point: ' + str(v))
    print(sigmoid(m * float(v[2]) - float(v[3]) + b))

    pass

def plot_data_line(d, w0, w1, w2, color='c', marker='solid', show=False):
    plot_data(d)
    plt.axline(xy1=(0, -w0 / w2), xy2=(-w0 / w1, 0), color=color, linestyle=marker)

    plt.xlim(2.9, 7.1)
    plt.ylim(0.9, 2.6)

    if show:
        plt.show()

    pass

def actualPoint(flower):
    if flower == 'versicolor':
        return 0.0
    else:
        return 1.0
    
def meanSquaredError(d, w0, w1, w2, color='c', marker='-', plot=True):
    if not plot:
        plot_data_line(d, w0, w1, w2, color=color, marker=marker)

    E = 0.0
    for r in d:
        x1 = float(r[2])
        x2 = float(r[3])

        z = w0 + w1 * x1 + w2 * x2
        sigma = sigmoid(z)

        v = actualPoint(r[4])

        E += math.pow(v - sigma, 2)

    return E / (2.0 * len(d))


def meanSquaredError_2Points(d, w00, w10, w20, w01, w11, w21):
    meanSquaredError0 = meanSquaredError(d, w00, w10, w20, 'c', 'solid')
    meanSquaredError1 = meanSquaredError(d, w01, w11, w21, 'r', 'dashed')
    plt.show()
    return 0 

def gradient_sum(d, w0, w1, w2, plot=True):
    
    # f(z) = 1/(1 + e-z)
    # f'(z) = (e-z)/(1 + e-z)2
    # z = mx - y + b

    # f'(x) = m(e(y - mx - b))/(1 + e(y - mx - b))2
    # f'(y) = -(e(y - mx - b))/(1 + e(y - mx - b))2


    if plot:
        plot_data_line(d, w0, w1, w2)

    grad_w0 = 0.0
    grad_w1 = 0.0
    grad_w2 = 0.0

    for r in d:
        x1 = float(r[2])
        x2 = float(r[3])

        z = w0 + w1 * x1 + w2 * x2

        v = actualPoint(r[4])

        sigma = sigmoid(z)

        d_sigma = math.exp(-z)/math.pow((1 + math.exp(-z)), 2)

        df_dz = 2.0 * (v - sigma) * -d_sigma

        grad_w0 += df_dz * 1
        grad_w1 += df_dz * x1
        grad_w2 += df_dz * x2

    grad_w0 /= len(d)
    grad_w1 /= len(d)
    grad_w2 /= len(d)

    #
    step_size = -10.0
    if plot:
        plot_data_line(d, w0 + grad_w0 * step_size, w1 + grad_w1 * step_size, w2 + grad_w2 * step_size, 'r')
        plt.show()

    return [grad_w0, grad_w1, grad_w2]

def gradient_decent(d, w0, w1, w2, plot_learning_curve=False):
    step = 0.1
    stopping_criteria = 0.01
    learning_curve = []

    while True:
        g = gradient_sum(d, w0, w1, w2, False)
        w0 -= g[0] * step
        w1 -= g[1] * step
        w2 -= g[2] * step

        norm = math.sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2])
        learning_curve.append(norm)

        if norm < stopping_criteria:
            if plot_learning_curve:
                return [[w0, w1, w2], learning_curve]
            return [w0, w1, w2]


def show_gradient_decent(d, w0, w1, w2, show_curve=True):
    plot_data_line(d, w0, w1, w2)

    if show_curve:
        line, curve = gradient_decent(d, w0, w1, w2, True)
        plot_data_line(d, line[0], line[1], line[2], 'r')
        plt.show()

        plt.plot(range(len(curve)), curve, 'k')
        plt.xlabel('Iterations')
        plt.ylabel('Norm of Gradient')
        plt.show()
        return line, curve

    else:
        line = gradient_decent(d, w0, w1, w2)
        plot_data_line(d, line[0], line[1], line[2], 'r')
        plt.show()
        return line


def random_gradient_decent(d):
    np.random.seed(1234)
    w = np.random.uniform(-5, 5, 3)
    return show_gradient_optimized(d, w[0], w[1], w[2], True)
    
