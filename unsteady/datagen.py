import numpy as np
from matplotlib import pyplot as plt
from pyDOE import lhs
from functional import setSeed


def dataGen(numIn, numOut, numCL, numOB, numIc, lowB, uppB, cyldCoord, cyldRadius, randSeed=42):
    """
    :param numIn: Number of inlet points, default is 1000
    :param numOut: Number of outlet points, default is 1000
    :param numCL: Number of collocation points, default is 30000, plus, a 1/5 refined pts will be added before cylinder
    :param numOB: Number of obstacle points, default is 1000 for each (cylinder, upper and lower wall)
    :param numIc: Number of initial points, default is 1000
    :param lowB: Order in (x,y,t).
    :param uppB: Order in (x,y,t).
    :param cyldCoord: Coordinate of cylinder, default is [0.2,0.2]
    :param cyldRadius: Radius of cylinder, default is 0.5
    :param randSeed: Global random seed, default is 42, the answer to the universe.
    :return: A list of [[collocationPtsX, ...], [inletPtsX, ...], [outletPtsX, ...], [obstaclePtsX, ...]]
             order in (X,Y,T,U,V), U,V for inlet only.
    """

    setSeed(randSeed)

    # INLET (x,y,t,u,v)
    maxU = 1.5
    T = uppB[2]*2
    inletPts = lowB + [0., uppB[1], uppB[2]] * lhs(3, numIn)
    # inletU = 4 * maxU * inletPts[:, 1] * (uppB[1] - inletPts[:, 1]) / uppB[1] ** 2
    inletU = 4 * maxU * inletPts[:, 1] * (0.41 - inletPts[:, 1]) / (0.41 ** 2) * (
           np.sin(2 * 3.1415927 * inletPts[:, 2] / T + 3 * 3.1415927 / 2) + 1.0)
    inletV = np.zeros_like(inletU)
    inletPts = np.vstack((inletPts[:, 0], inletPts[:, 1], inletPts[:, 2], inletU, inletV)).T  # some annoying stacking

    # OUTLET (x,y,t)
    outPts = [uppB[0], lowB[1], lowB[2]] + [0, uppB[1], uppB[2]] * lhs(3, numOut)

    # OBSTACLE (x,y,t)
    lowerWall = lowB + [uppB[0], 0., uppB[2]] * lhs(3, numOB)
    upperWall = [lowB[0], uppB[1], lowB[2]] + [uppB[0], 0., uppB[2]] * lhs(3, numOB)

    cyldTheta, cyldT = ([0.0, lowB[2]] + [2 * np.pi, uppB[2]] * lhs(2, numOB)).T
    cyldX = np.multiply(cyldRadius, np.cos(cyldTheta)) + cyldCoord[0]
    cyldY = np.multiply(cyldRadius, np.sin(cyldTheta)) + cyldCoord[1]
    cyldPts = np.vstack((cyldX, cyldY, cyldT)).T  # another one
    obstaclePts = np.concatenate((lowerWall, upperWall, cyldPts), 0)

    # Initial (x,y,t)
    icPts = lowB + [uppB[0], uppB[1], 0] * lhs(3, numIc)
    icPts = obstacleDel(icPts, cyldCoord, cyldRadius)

    # Collocation (x,y,t)
    colPts = lowB + uppB * lhs(3, numCL)
    refinedColPts = lowB + [cyldCoord[0], uppB[1], uppB[2]] * lhs(3, int(numCL / 5))
    colPts = np.concatenate((colPts, refinedColPts), 0)
    colPts = obstacleDel(colPts, cyldCoord, cyldRadius)
    colPts = np.concatenate((colPts, obstaclePts, inletPts[:, 0:3], outPts), 0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(inletPts[:, 1:2], inletPts[:, 2:3], inletPts[:, 3:4], marker='o', alpha=0.1, s=2, color='blue')
    ax.set_xlabel('y axis')
    ax.set_ylabel('t axis')
    ax.set_zlabel('v axis')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(inletPts[:, 0], inletPts[:, 1], inletPts[:, 2], marker='o', alpha=0.1, s=2, color='green')
    ax.scatter(outPts[:, 0], outPts[:, 1], outPts[:, 2], marker='o', alpha=0.1, s=2, color='orange')
    ax.scatter(obstaclePts[:, 0], obstaclePts[:, 1], obstaclePts[:, 2], marker='o', alpha=0.1, s=2, color='blue')
    ax.scatter(icPts[:, 0], icPts[:, 1], icPts[:, 2], marker='x', alpha=0.1, s=2, color='red')

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('T axis')
    plt.show()

    return [colPts, inletPts, outPts, obstaclePts, icPts]


def obstacleDel(pts, cyldCoord, cyldRadius):
    dst = np.array([((xy[0] - cyldCoord[0]) ** 2 + (xy[1] - cyldCoord[1]) ** 2) ** 0.5 for xy in pts])
    return pts[dst > cyldRadius, :]
