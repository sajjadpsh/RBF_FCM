import matplotlib.pyplot as plt


def draw(test, tresult, wrongs, centers, nc):
    xData = [[] for i in range(nc)] #x
    yData = [[] for i in range(nc)] #
    xc = []
    yc = []
    xw = []
    yw = []

    for i in range(0, test.shape[0]):
        if wrongs[i] == 0:
            xData[int(tresult[i]) - 1].append(test[i][0])
            yData[int(tresult[i]) - 1].append(test[i][1])
        else:
            xw.append(test[i][0])
            yw.append(test[i][1])

    for i in range(0, centers.shape[0]):
        xc.append(centers[i, 0])
        yc.append(centers[i, 1])

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
              '#17becf']
    for i in range(nc):
        for j in range(len(xData[i])):
            plt.scatter(xData[i][j], yData[i][j], s=10, c=colors[i])
    plt.scatter(xc, yc, s=30, c="b")
    plt.scatter(xw, yw, s=20, c="r", marker='x')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def draw2(data, result, nc):
    xData = [[] for i in range(nc)]
    yData = [[] for i in range(nc)]

    for i in range(0, data.shape[0]):
        xData[int(result[i])].append(data[i][0])
        yData[int(result[i])].append(data[i][1])

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
              '#17becf']
    for i in range(nc):
        for j in range(len(xData[i])):
            plt.scatter(xData[i][j], yData[i][j], s=10, c=colors[i])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()