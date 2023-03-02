import argparse
from datagen import dataGen
from functional import setSeed
from unsteady.model import NSModel

parser = argparse.ArgumentParser()
parser.add_argument('--layer', help='number of layers', type=int, default=7)
parser.add_argument('--neurons', help='number of neurons per layer', type=int, default=50)
parser.add_argument('--numIn', help='number of init points pper layer', type=int, default=4000)
parser.add_argument('--numOut', help='number of boundary points', type=int, default=4000)
parser.add_argument('--numCL', help='number of collocation points', type=int, default=80000)
parser.add_argument('--numOB', help='number of obstacle points', type=int, default=4000)
parser.add_argument('--numIC', help='number of initial points', type=int, default=4000)
parser.add_argument('--AEpoch', help='Number of ADAM epochs', type=int, default=10000)
parser.add_argument('--LEpoch', help='Number of LBFGS epochs', type=int, default=30000)
parser.add_argument('--act', help='Activation function', type=str, default='tanh')
parser.add_argument('--save', help='save model', type=bool, default=True)
parser.add_argument('--record', help='Make Tensorboard record', type=bool, default=True)
parser.add_argument('--seed', help='Random seed', type=int, default=42)


def main():
    args = parser.parse_args()
    print(args)

    setSeed(args.seed)

    lowerBound = [0, 0, 0]
    upperBound = [1.1, 0.41, 0.5]
    cyldCoord = [0.2, 0.2]
    cyldRadius = 0.05

    bound = [lowerBound, upperBound]
    nnPara = [args.layer, args.neurons, args.act]
    iterPara = [args.AEpoch, args.LEpoch]

    pts = dataGen(args.numIn, args.numOut, args.numCL, args.numOB, args.numIC, lowerBound, upperBound, cyldCoord, cyldRadius, 42)
    model = NSModel(pts, bound, nnPara, iterPara, args.save, args.record, args.seed)
    # model.loadFCModel("./models/model.pt")
    model.train()
    model.inference()


if __name__ == "__main__":
    main()
