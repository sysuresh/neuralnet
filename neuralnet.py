import numpy as np
from scipy import optimize 

class NeNet(object):
    def __init__(self, insize, outsize, hidsize):
        self.inputlayersize = insize
        self.outputlayersize = outsize
        self.hiddenlayersize = hidsize

        self.Winput = np.random.randn(self.inputlayersize, self.hiddenlayersize)
        self.Woutput = np.random.randn(self.hiddenlayersize, self.outputlayersize)   

    def fprop(self, INPUT):
        self.hz1 = np.dot(INPUT, self.Winput)
        self.ha2 = self.sig(self.hz1)
        self.hz2 = np.dot(self.ha2, self.Woutput)
        ybar = self.sig(self.hz2)
        return ybar
    def getGradient(self, INPUT, OUTPUT):
        step = 1
        self.dWinput, self.dWoutput = self.dcost(INPUT, OUTPUT)
        return np.concatenate((self.dWinput.ravel(), self.dWoutput.ravel()))
    def approximateGradient(self, INPUT, OUTPUT):
        weightVector = self.getWeightVector()
        estgrad = np.zeros(weightVector.shape)
        pert = np.zeros(weightVector.shape)
        epsilon = 1e-4

        for p in range(len(weightVector)):
            pert[p] = epsilon
            self.setWeightVector(weightVector + pert)
            loss2 = self.cost(INPUT, OUTPUT)
            self.setWeightVector(weightVector - pert)
            loss1 = self.cost(INPUT, OUTPUT)

            estgrad[p] = (loss2 - loss1)/(2*epsilon)

            pert[p] = 0
        
        self.setWeightVector(self.getWeightVector())
        return estgrad #ideally less than 10^-8

    def cost(self, INPUT, OUTPUT):
        y = OUTPUT
        self.ybar = self.fprop(INPUT)
        return 0.5*sum((y-self.ybar)**2)

    def dcost(self, INPUT, OUTPUT): #return dWinput, dWoutput towards cost function minimum
        y = OUTPUT
        self.ybar = self.fprop(INPUT)

        er3 = np.multiply(self.ybar-y, self.dsig(self.hz2))
        dWoutput = np.dot(self.ha2.T, er3)

        er2 = np.dot(er3, self.Woutput.T)*self.dsig(self.hz1)
        dWinput = np.dot(INPUT.T, er2)

        return dWinput, dWoutput
     
    def sig(self, z):
        z = np.clip(z, -500, 500)
        return 1/(1+np.exp(-z))
    def dsig(self, z):
        z = np.clip(z, -100, 100)        
        return (np.exp(-z)/(1+np.exp(-z))**2)
    
    def getWeightVector(self):
        return np.concatenate((self.Winput.ravel(), self.Woutput.ravel()))
    
    def setWeightVector(self, pvector):
        Winputstart = 0
        Winputend = self.hiddenlayersize*self.inputlayersize
        self.Winput = np.reshape(pvector[Winputstart:Winputend], (self.inputlayersize, self.hiddenlayersize))
        Woutputstart = Winputend
        Woutputend = Woutputstart + self.hiddenlayersize*self.outputlayersize
        self.Woutput = np.reshape(pvector[Woutputstart:Woutputend], (self.hiddenlayersize, self.outputlayersize))
class NeNetTrainer(object):
    def __init__(self, N):
        self.N = N
    def costwrapper(self, parameters, INPUT, OUTPUT):
        self.N.setWeightVector(parameters)
        cost = self.N.cost(INPUT, OUTPUT)
        grad = self.N.getGradient(INPUT, OUTPUT)
        return cost, grad
    def callback(self, params):
        self.N.setWeightVector(params)
        self.COST.append(self.N.cost(self.X, self.y))
    def train(self, INPUT, OUTPUT):
        self.X = INPUT
        self. y = OUTPUT

        self.COST = []

        params0 = self.N.getWeightVector()

        options = {'maxiter': 200, 'disp' : False}
        _res = optimize.minimize(self.costwrapper, params0, jac = True, method = 'BFGS', args = (INPUT, OUTPUT), options=options, callback = self.callback)
        
        #update weights
        self.N.setWeightVector(_res.x)
        self.optimizationResults = _res


#simple XOR function as example
X = np.array(([0, 0], [0, 1], [1, 0], [1,1]), dtype = int)
X = X/np.amax(X, axis=0)
y = np.array(([0], [1], [1], [0]), dtype = int)
y = y/np.amax(y, axis=0)
while True:
    N = NeNet(2, 1, 2)
    trainer = NeNetTrainer(N)
    trainer.train(X, y)
    result = trainer.optimizationResults
    if result.fun < 0.1:
        print(N.fprop(X)) 
        print("Final cost function value: " + str(round(result.fun, 9)))
        break
