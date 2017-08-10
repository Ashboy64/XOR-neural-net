import numpy as np

def sigmoid(x,derivative=False):
    if(derivative==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

np.random.seed(1)

weights_1 = np.random.randn(2,3)
weights_2 = np.random.randn(1,3)

training = np.array([[np.array([1,1,1]).reshape(1,-1),0],
                    [np.array([0,1,1]).reshape(1,-1),1],
                    [np.array([1,0,1]).reshape(1,-1),1],
                    [np.array([0,0,1]).reshape(1,-1),0]])


for x in range(10000):
    for iter in xrange(training.shape[0]):
#forwardPropagation:
        a_layer1 = training[iter][0]
        z_layer2 = np.dot(weights_1,a_layer1.reshape(-1, 1))
        a_layer2_noBias = sigmoid(z_layer2)
        a_layer2 = np.array([a_layer2_noBias[0][0], a_layer2_noBias[1][0],float(1)]).reshape(1,-1)
        z_layer3 = np.dot(weights_2,a_layer2.reshape(-1, 1))
        a_layer3 = sigmoid(z_layer3)
        hypothesis_theta = a_layer3

#backPropagation:
        delta_layer3 = (a_layer3 - training[iter][1] ) * sigmoid(a_layer3, derivative = True)
        #print delta_layer3
        #print z_layer2
        delta_layer2 = np.dot(weights_2.T, delta_layer3)*sigmoid(a_layer2, derivative = True)
        Delta_layer1 = np.dot(delta_layer2 , a_layer1.T)
        Delta_layer2 = np.dot(delta_layer3, a_layer2)
        update1 = Delta_layer1.T
        update2 = Delta_layer2
        weights_1 = weights_1 - update1
        weights_2 = weights_2 - update2


x = np.array([1,0, 1])
a_layer1 = x
z_layer2 = np.dot(weights_1,a_layer1.reshape(-1, 1))
a_layer2_noBias = sigmoid(z_layer2)
a_layer2 = np.array([a_layer2_noBias[0][0], a_layer2_noBias[1][0],float(1)]).reshape(1,-1)
z_layer3 = np.dot(weights_2,a_layer2.reshape(-1, 1))
a_layer3 = sigmoid(z_layer3)
print a_layer3


