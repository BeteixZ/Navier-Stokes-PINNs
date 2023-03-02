import os
import shutil

import torch
import torch.nn as nn
import numpy as np
import scipy.io as sp
from functools import partial

from matplotlib import pyplot as plt
from pyDOE import lhs
from torch.autograd import Variable
import time
from torch import sin, exp
from numpy import pi
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from functional import derivative, initWeights, setSeed, postProcess


class FCModel(nn.Module):
    def __init__(self, layer: int = 8, neurons: int = 40, act: str = 'tanh'):
        # Input layer
        super(FCModel, self).__init__()
        self.linear_in = nn.Linear(3, neurons)  # (x,y,t)
        self.linear_out = nn.Linear(neurons, 5)
        self.layers = nn.ModuleList(
            [nn.Linear(neurons, neurons) for i in range(layer)]
        )
        # Activation function
        if act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'gelu':
            self.act = nn.GELU()
        elif act == 'mish':
            self.act = nn.Mish()
        else:
            self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_in(x)
        for layer in self.layers:
            x = self.act(layer(x))
        x = self.linear_out(x)
        return x


class NSModel:
    def __init__(self, pts, bound, nnPara, iterPara, save, record, randSeed):
        setSeed(randSeed)
        self.device = "cuda:0"  # force to use GPU
        self.colPts = pts[0]
        self.inletPts = pts[1]
        self.outPts = pts[2]
        self.obstaclePts = pts[3]
        self.icPts = pts[4]
        self.lowB = bound[0]
        self.uppB = bound[1]
        self.model = FCModel(nnPara[0], nnPara[1], nnPara[2]).to(self.device)

        self.mseClV = 0
        self.mseInletV = 0
        self.mseOutletV = 0
        self.mseObsV = 0

        self.rho = 1
        self.mu = 0.001

        self.iterADAM = iterPara[0]
        self.iterLBFGS = iterPara[1]
        self.__nowIter = 0
        self.__nowLoss = 0

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.optimizerLFBGS = torch.optim.LBFGS(self.model.parameters(),
                                                max_iter=self.iterLBFGS,
                                                tolerance_change=0,
                                                line_search_fn="strong_wolfe")

        self.record = record
        self.save = save
        self.writer = self.__summaryWriter(nnPara[0], nnPara[1], nnPara[2])
        self.avgEpochTime = time.time()
        self.model.apply(initWeights)

    def __summaryWriter(self, layer, neurons, act):
        if self.record:
            return SummaryWriter(comment="")  # TODO

    def loadFCModel(self, dir):
        self.model.load_state_dict(torch.load(dir))

    def variablize(self, pts, reqGrad=None):
        if reqGrad is None:
            reqGrad = [True, True, False]
        colX = Variable(torch.from_numpy(pts[:, 0].astype(np.float32)), requires_grad=reqGrad[0]).to(self.device)
        colY = Variable(torch.from_numpy(pts[:, 1].astype(np.float32)), requires_grad=reqGrad[1]).to(self.device)
        colT = Variable(torch.from_numpy(pts[:, 2].astype(np.float32)), requires_grad=reqGrad[2]).to(self.device)
        return colX, colY, colT

    def __uvp(self, X, Y, T):
        modelOut = self.model(torch.stack((X, Y, T), dim=1))
        psi = modelOut[:, 0]
        u = derivative(psi, Y)
        v = -derivative(psi, X)
        p = modelOut[:, 1]
        return u, v, p

    def __mseCollocation(self):
        '''

        :return:
        '''
        colX, colY, colT = self.variablize(self.colPts, reqGrad=[True, True, True])
        modelOut = self.model(torch.stack((colX, colY, colT), dim=1))
        psi = modelOut[:, 0]
        p = modelOut[:, 1]
        s11 = modelOut[:, 2]
        s22 = modelOut[:, 3]
        s12 = modelOut[:, 4]
        u = derivative(psi, colY)  # and this
        v = -derivative(psi, colX)

        s11_x = derivative(s11, colX)
        s12_y = derivative(s12, colY)
        s22_y = derivative(s22, colY)
        s12_x = derivative(s12, colX)

        # Plane stress problem
        u_x = derivative(u, colX)
        u_y = derivative(u, colY)
        u_t = derivative(u, colT)

        v_x = derivative(v, colX)
        v_y = derivative(v, colY)
        v_t = derivative(v, colT)

        # f_u = Sxx_x + Sxy_y
        f_u = self.rho * (u * u_x + v * u_y + u_t) - s11_x - s12_y
        f_v = self.rho * (u * v_x + v * v_y + v_t) - s12_x - s22_y

        # f_mass = u_x + v_y
        f_s11 = -p + 2 * self.mu * u_x - s11
        f_s22 = -p + 2 * self.mu * v_y - s22
        f_s12 = self.mu * (u_y + v_x) - s12

        f_p = p + (s11 + s22) / 2

        return torch.mean(f_u ** 2) + torch.mean(f_v ** 2) + \
            torch.mean(f_s11 ** 2) + torch.mean(f_s22 ** 2) + \
            torch.mean(f_s12 ** 2) + torch.mean(f_p ** 2)

    def __mseInlet(self):
        inletX, inletY, inletT = self.variablize(self.inletPts)
        modelOut = self.model(torch.stack((inletX, inletY, inletT), dim=1))  # TODO
        psi = modelOut[:, 0]
        v = -derivative(psi, inletX)
        u = derivative(psi, inletY)
        uReal = Variable(torch.from_numpy(self.inletPts[:, 3].astype(np.float32)), requires_grad=False).to(self.device)
        return torch.mean((u - uReal) ** 2) + torch.mean(v ** 2)

    def __mseOutlet(self):
        outletX, outletY, outletT = self.variablize(self.outPts)
        return torch.mean(self.model(torch.stack((outletX, outletY, outletT), dim=1))[:, 1] ** 2)

    def __mseObstacle(self):
        obstacleX, obstacleY, obstacleT = self.variablize(self.obstaclePts)
        modelOut = self.model(torch.stack((obstacleX, obstacleY, obstacleT), dim=1))
        psi = modelOut[:, 0]
        v = -derivative(psi, obstacleX)
        u = derivative(psi, obstacleY)
        return torch.mean(u ** 2) + torch.mean(v ** 2)

    def __mseIC(self):
        icX, icY, icT = self.variablize(self.icPts)
        modelOut = self.model(torch.stack((icX, icY, icT), dim=1))
        psi = modelOut[:, 0]
        u = derivative(psi, icY)
        v = -derivative(psi, icX)
        p = modelOut[:, 1]
        return torch.mean(u ** 2) + torch.mean(v ** 2) + torch.mean(p ** 2)

    def __closure(self):
        self.avgEpochTime = time.time()
        self.__nowIter += 1
        self.optimizer.zero_grad()
        mseCl = self.__mseCollocation()
        mseIn = self.__mseInlet()
        mseOut = self.__mseOutlet()
        mseObs = self.__mseObstacle()
        mseIC = self.__mseIC()
        loss = mseCl + 3 * mseIn + 6 * mseOut + 3 * mseObs + 2 * mseIC
        self.__nowLoss = loss.item()
        loss.backward(retain_graph=False)

        if self.record:
            self.writer.add_scalar('Loss', loss.detach().cpu().numpy(), self.__nowIter)
            self.writer.add_scalars('MSE', {'MSE_f': mseCl.detach().cpu().numpy(),
                                            'MSE_in': mseIn.detach().cpu().numpy(),
                                            'MSE_out': mseOut.detach().cpu().numpy(),
                                            'MSE_obstacle': mseObs.detach().cpu().numpy(),
                                            'MSE_ic': mseIC.detach().cpu().numpy()}, self.__nowIter)

        if self.__nowIter % 10 == 0:
            print("Iter: {}, AvgT: {:.2f}, loss: {:.6f}, Cl: {:.6f}, In: {:.6f}, Out:{:.6f}, Obs:{:.6f}, IC:{:.6f}"
                  .format(self.__nowIter,
                          time.time() - self.avgEpochTime,
                          loss.item(),
                          mseCl,
                          mseIn,
                          mseOut,
                          mseObs,
                          mseIC))
        return loss

    def train(self):
        timeStart = time.time()
        scheduler = StepLR(self.optimizer, step_size=2000, gamma=0.5)
        print("Start training: ADAM")
        for i in range(self.iterADAM):
            self.optimizer.step(self.__closure)
            scheduler.step()

        torch.cuda.empty_cache()
        print("Start training: L-BFGS")
        self.optimizer = torch.optim.LBFGS(self.model.parameters(),
                                           lr=1,
                                           max_iter=self.iterLBFGS,
                                           history_size=100,
                                           tolerance_change=0,
                                           line_search_fn="strong_wolfe")
        self.optimizer.step(self.__closure)
        print('Total time cost: ', time.time() - timeStart, 's')

        if self.save:
            torch.save(self.model.state_dict(),
                       './models/' + "dortmund-2d-2-unsteay-model" + '.pt')

    def inference(self):  # not change too much
        # tFront = np.linspace(0, self.uppB[2], 100)
        # xFront = np.zeros_like(tFront)
        # xFront.fill(0.15)
        # yFront = np.zeros_like(tFront)
        # yFront.fill(0.20)
        # tFront = tFront.flatten()[:, None]
        # xFront = xFront.flatten()[:, None]
        # yFront = yFront.flatten()[:, None]

        # x_frontT = Variable(torch.from_numpy(xFront.astype(np.float32)), requires_grad=True).to(self.device)
        # y_frontT = Variable(torch.from_numpy(yFront.astype(np.float32)), requires_grad=True).to(self.device)
        # t_frontT = Variable(torch.from_numpy(tFront.astype(np.float32)), requires_grad=False).to(self.device)
        self.model.eval()

        # p with t
        # _, _, pPred = self.__uvp(x_frontT[:, 0], y_frontT[:, 0], t_frontT[:, 0])
        # pPred = pPred.data.cpu().numpy()
        # plt.plot(tFront, pPred)
        # plt.show()

        N_t = 51
        xStar = np.linspace(self.lowB[0], self.uppB[0], 401)
        yStar = np.linspace(self.lowB[1], self.uppB[1], 161)
        xStar, yStar = np.meshgrid(xStar, yStar)
        xStar = xStar.flatten()[:, None]
        yStar = yStar.flatten()[:, None]
        dst = ((xStar - 0.2) ** 2 + (yStar - 0.2) ** 2) ** 0.5  # consider change this: TODO
        xStar = xStar[dst >= 0.05]
        yStar = yStar[dst >= 0.05]
        # xStar = xStar.flatten()[:, None]
        # yStar = yStar.flatten()[:, None]

        xStarT = Variable(torch.from_numpy(xStar.astype(np.float32)), requires_grad=True).to(self.device)
        yStarT = Variable(torch.from_numpy(yStar.astype(np.float32)), requires_grad=True).to(self.device)

        shutil.rmtree('./output', ignore_errors=True)
        os.makedirs('./output')
        for i in range(N_t):
            tStar = np.zeros((xStar.size, 1))
            tStar.fill(i * self.uppB[2] / (N_t - 1))

            tStarT = Variable(torch.from_numpy(tStar.astype(np.float32)), requires_grad=False).to(self.device)

            uPred, vPred, pPred = self.__uvp(xStarT, yStarT, tStarT[:, 0])
            uPred = uPred.data.cpu().numpy()
            vPred = vPred.data.cpu().numpy()
            pPred = pPred.data.cpu().numpy()
            field = [xStar, yStar, tStar, uPred, vPred, pPred]

            postProcess(xmin=self.lowB[0], xmax=self.uppB[0], ymin=self.lowB[1], ymax=self.uppB[1], field=field, s=2,
                        num=i, tstep=self.uppB[2] / (N_t - 1))
