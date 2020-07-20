import csv
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import random

FILENAME = '2clstrain1200.csv'
M = 13
RADUIS = .2


def G_matrix_exponential(X_vector, V_vector, C_vector, landa):
    g = []

    for i in range(len(X_vector)):
        for j in range(len(V_vector)):
            g.append(np.exp(-landa * np.power(X_vector[i] - V_vector[j], 2) * C_vector[j]))
    return np.array(g).reshape([len(X_vector), len(V_vector)])


############################ FCM ###########################
def fcm(data, data_train, Y_train, data_test, Y_test, classes_count=2, n_clusters=1, landa=.1, n_init=30, m=2,
        max_iter=300, tol=1e-16):
    um = None
    min_cost = np.inf
    for iter_init in range(n_init):
        centers = data[np.random.choice(
            data.shape[0], size=n_clusters, replace=False
        ), :]

        dist = np.fmax(
            cdist(centers, data, metric='sqeuclidean'),
            np.finfo(np.float64).eps
        )
        for iter1 in range(max_iter):

            u = (1 / dist) ** (1 / (m - 1))
            um = (u / u.sum(axis=0)) ** m

            prev_centers = centers
            centers = um.dot(data) / um.sum(axis=1)[:, None]

            dist = cdist(centers, data, metric='sqeuclidean')

            if np.linalg.norm(centers - prev_centers) < tol:
                break

        cost = np.sum(um * dist)
        if cost < min_cost:
            min_cost = cost
            min_centers = centers
            mem = um.argmax(axis=0)
    # plt.plot(data[:, 0], data[:, 1], 'go', min_centers[:, 0], min_centers[:, 1], 'ys')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
              '#17becf']
    # for i in range(len(min_centers)):
    #     for j in range(len(data[i])):
    #         plt.scatter(data[i][j], data[i][j], s=10, c=colors[i])
    plt.scatter(data[:, 0], data[:, 1], s=30, c="g", marker='o')
    plt.scatter(min_centers[:, 0], min_centers[:, 1], s=20, c="y", marker='s')
    plt.show()
    C = calculate_covariance_mat(data, min_centers, um, m)
    G = np.zeros([len(data_train), len(min_centers)])
    for k in range(len(data_train)):
        for i in range(len(min_centers)):
            G[k][i] += np.exp(-landa * np.matmul(
                np.matmul(np.transpose(np.array(data_train[k]) - np.array(min_centers[i])), np.linalg.inv(C[i])),
                np.array(np.array(data_train[k]) - np.array(min_centers[i]))))
    try:
        W = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(G), G)), np.transpose(G)),
                      generate_Y(len(data_train), Y_train, classes_count))
    except np.linalg.LinAlgError:
        W = np.matmul(np.matmul(np.random.rand([len(min_centers), len(min_centers)]), np.transpose(G)),
                      generate_Y(len(data_train), Y_train, classes_count))
    y_hat = np.argmax(np.matmul(G, W), axis=1)
    acc = accuracy(Y_train, y_hat, len(data))
    print("train data accuracy: ", acc)
    G = np.zeros([len(data_test), len(min_centers)])
    for k in range(len(data_test)):
        for i in range(len(min_centers)):
            G[k][i] += np.exp(-landa * np.matmul(
                np.matmul(np.transpose(np.array(data_test[k]) - np.array(min_centers[i])), np.linalg.inv(C[i])),
                np.array(np.array(data_test[k]) - np.array(min_centers[i]))))
    y_hat = np.argmax(np.matmul(G, W), axis=1)
    g1_x, g1_y, g2_x, g2_y = [], [], [], []
    for i in range(len(y_hat)):
        if y_hat[i] == Y_test[i]:
            g1_x.append(data_test[i][0])
            g1_y.append(data_test[i][1])
        else:
            g2_x.append(data_test[i][0])
            g2_y.append(data_test[i][1])
    plt.plot(data[:, 0], data[:, 1], 'c*', g1_x, g1_y, 'go', g2_x, g2_y, 'ro')
    plt.show()

    acc = accuracy(Y_test, y_hat, len(data))
    print("test data accuracy: ", acc)
    return min_centers


def generate_Y(x_d, y_vector, classes_count):
    Y = []
    for i in range(x_d):
        y_i = [0 for i in range(classes_count)]
        y_i[y_vector[i]] = 1
        Y.append(y_i)
    return np.array(Y).reshape([x_d, classes_count])


def calculate_covariance_mat(data, centers, U, m):
    C = []
    centers = np.array(centers)
    for i in range(len(centers)):
        shape = (2, 2)
        divided_sum = np.zeros(shape)
        divisor_sum = .0
        for k in range(len(data)):
            divided_sum += np.power(U[i][k], m) * np.multiply(
                np.array([data[k] - centers[i]]), np.transpose(
                    np.array([data[k] - centers[i]])
                )
            )
            divisor_sum += np.power(U[i][k], m)
        C.append(divided_sum / divisor_sum)
    return C


def accuracy(y, y_hat, n):
    return 1 - (np.sum(np.abs(np.sign(np.subtract(y, y_hat)))) / n)


############################ FCM ###########################

def plot_inputs(filename):
    inputs = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if row != []:
                inputs.append(row)
    csv_file.close()

    x1 = []
    x2 = []
    y1 = []
    y2 = []
    for item in inputs:
        if float(item[2]) == 1.0:
            x1.append(float(item[0]))
            y1.append(float(item[1]))
        else:
            x2.append(float(item[0]))
            y2.append(float(item[1]))
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
    #           '#17becf']
    # for i in range(len(colors)):
    # for j in range(len(item[0])):
    #     plt.scatter(x1, y1, s=10, c=colors[i])
    plt.scatter(x1, y1, s=30, c="b")
    plt.scatter(x2, y2, s=20, c="c", marker='x')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    # plt.plot(x1, y1, 'y^', x2, y2, 'ro')
    # plt.show()


plot_inputs(FILENAME)

INPUT = []
Y = []


def read_data(filename):
    with  open(filename) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if row != []:
                INPUT.append([float(row[0]), float(row[1])])
                Y.append(0 if float(row[2]) == 1.0 else 1)
    csv_file.close()


read_data(FILENAME)
z = list(zip(np.array(INPUT), np.array(Y)))
random.shuffle(z)
INPUT, Y = zip(*z)
x_train = INPUT[:int(.7 * len(INPUT))]
x_test = INPUT[int(.7 * len(INPUT)):len(INPUT)]
y_train = Y[:int(.7 * len(INPUT))]
y_test = Y[int(.7 * len(INPUT)):len(INPUT)]
fcm(np.array(INPUT), x_train, y_train, x_test, y_test, 2, M)
# trainRBF(INPUT,Y,.1,fcm(np.array(INPUT),M),2,M)
