import numpy as np
import torch
import torch.nn as nn 
import torch.optim as optim
import matplotlib.pyplot as plt

#####################################################################################
###########################            Dataset           ############################
#####################################################################################

class NoisyDataset:
    def __init__(self, features, func, domain = np.array([-3,3]), w_star = np.array([0,-5,2,1,0.05])):
        #Declare domain and 
        self.domain = domain
        self.w_star = w_star.T
        self.features = features
        self.func = func

        #declare params
        self.sizes = None
        self.seeds = None
        self.sigmas = None

        #declare outputs (training and validation sets)
        self.x_tr = None 
        self.X_tr = None 
        self.y_tr = None 
        self.x_val = None 
        self.X_val = None 
        self.y_val = None

    def get(self, sizes = np.array([500,500]), seeds = np.array([0,1]), sigmas = np.array([.5,.5])):
        #check to see if the computation has been done and do if not
        if np.all(sizes != self.sizes) or np.all(seeds != self.seeds) or np.all(sigmas == self.sigmas):
            self.sizes = sizes
            self.seeds = seeds
            self.sigmas = sigmas

            for k in range(2):
                random_state = np.random.RandomState(self.seeds[k])

                x = random_state.uniform(self.domain[0], self.domain[1], (self.sizes[k]))
                X = np.zeros((self.sizes[k], self.w_star.shape[0]))
                for i in range(self.sizes[k]):
                    X[i,0] = 1
                    for j in range(1, self.w_star.shape[0]):
                        X[i,j] = self.features(x,i,j)

                y = X.dot(self.w_star)
                if self.sigmas[k] > 0:
                    y += random_state.normal(0.0, self.sigmas[k], self.sizes[k])

                if k == 0:
                    self.x_tr, self.X_tr, self.y_tr = x, X, y
                else:
                    self.x_val, self.X_val, self.y_val = x, X, y
            
        return self.x_tr, self.X_tr, self.y_tr, self.x_val, self.X_val, self.y_val

    def saveAndDisplayDataset(self, plt, name = 'scatter.png'):
        x = np.linspace(self.domain[0], self.domain[1], 100)
        plt.title('Training and Validation Sets for F(x)')

        plt.subplot(1,2,1)
        plt.plot(x, self.func(self.w_star, x), label='Ground Truth')
        plt.scatter(self.x_tr, self.y_tr, color='orange', label='Training Set')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(x, self.func(self.w_star, x), label='Ground Truth')
        plt.scatter(self.x_val, self.y_val, color='green', label='Validation Set')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()

        plt.savefig(name)


#####################################################################################
###########################            Regression           #########################
#####################################################################################

class RegressionModel:
    #Init regression class
    #   dataset : NoisyDataset object
    #   func : function mapping w_star and linspace([a,b]) to func([a,b])
    #   dim : number of features in each feature vector (eg. 4 for degree 3 polynomial)
    #   verbose : displays all runtime data
    #   device : runs on device chosen (eg. 'cpu' or 'gpu')
    def __init__(self, dataset, func, dim=5, device='cpu'):
        #get parameters
        self.dataset = dataset
        self.func = func
        self.dim = dim
        self.verbose = False
        self.device = device

        #get training and validation sets
        self.x_tr = None
        self.X_tr = None 
        self.y_tr = None 
        self.x_val = None 
        self.X_val = None 
        self.y_val = None
        self.x_tr, self.X_tr, self.y_tr, self.x_val, self.X_val, self.y_val = dataset.get()

        #getb model params
        self.alpha = 0
        self.maxIts = 0
        self.model = None

    #Train the regresson model on the dataset data
    #   alpha : learning rate
    #   maxIts : max iterations in 
    def train(self, alpha=.001, maxIts=3000, verbose=True, run=True):
        if run or (not self.model or not self.alpha == alpha or not self.maxIts == maxIts):
            self.alpha = alpha
            self.maxIts = maxIts
            self.verbose = verbose

            #Get number of samples
            num_samples_tr = self.X_tr.shape[0]
            num_samples_val = self.X_val.shape[0]

            #Define model
            self.model = nn.Linear(self.dim, 1, bias=False)
            self.model = self.model.to(self.device)
            Loss = nn.MSELoss()
            optimizer = optim.SGD(self.model.parameters(), lr=self.alpha)
            if self.verbose: 
                with torch.no_grad(): print(f'Initial params:\t{self.model.weight.numpy()}')

            #Format data for model
            X_train = torch.from_numpy(self.X_tr.reshape((num_samples_tr, self.dim))).float().to(self.device)
            y_train = torch.from_numpy(self.y_tr.reshape((num_samples_tr, 1))).float().to(self.device)

            X_validation = torch.from_numpy(self.X_val.reshape((num_samples_val, self.dim))).float().to(self.device)
            y_validation = torch.from_numpy(self.y_val.reshape((num_samples_val, 1))).float().to(self.device)

            #train model
            self.losses_tr = []
            self.losses_val = []
            with torch.no_grad(): self.weights = self.model.weight.numpy()
            for step in range(maxIts):
                #Put model into training mode with zero grad
                self.model.train()
                optimizer.zero_grad()

                #Predict using the curent model
                y_pred_train = self.model(X_train)
                loss_train = Loss(y_pred_train, y_train)
                self.losses_tr.append(loss_train.item())
                if self.verbose: print(f'Step {step}: train loss - {loss_train}')

                #Compute grads and update parameters
                loss_train.backward()
                optimizer.step()

                #Evaluate on validation set
                self.model.eval()
                with torch.no_grad():
                    self.weights = np.concatenate((self.weights, self.model.weight.numpy()))
                    y_pred_validation = self.model(X_validation)
                    loss_validation = Loss(y_pred_validation, y_validation)
                    self.losses_val.append(loss_validation.item())
                if self.verbose: print(f'Step {step}: validation loss - {loss_validation}')

            #Save weights and prediction values from model
            with torch.no_grad():
                print(f'Final weight vector:\t{self.weights[-1]}')
                self.y_pred_tr = self.model(X_train).numpy()
                self.y_pred_val = self.model(X_validation).numpy()
                self.weights[1:].reshape((self.maxIts, self.dim))

    def saveAndDisplayResults(self, plt, name, domain = np.array([-3,3]), w_star = np.array([0,-5,2,1,0.05])):
        #Plot loss
        plt.subplot(2,3,1)
        x = np.linspace(domain[0],domain[1], self.maxIts)
        plt.plot(range(self.maxIts), self.losses_tr, color='blue', label='Training Loss')
        plt.plot(range(self.maxIts), self.losses_val, color='red', label='Validation Loss')
        plt.plot(range(self.maxIts), 0*x, color='black', label='Zero')
        plt.title(f'Polynomial Regression Loss for:\n loss_train = {self.losses_tr[-1]:.4f}, loss_val = {self.losses_val[-1]:.4f}')
        plt.xlabel('iterations')
        plt.ylabel('L(w)')
        plt.legend()

        #plot weights (reshape array to matrix)
        plt.subplot(2,3,2)
        for i in range(self.dim):
            plt.plot(range(self.maxIts), self.weights[:-1,i], label=f'weights[{i}]')
        plt.title('Weights over time')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
            
        #plot models
        plt.subplot(2,3,4)
        plt.plot(x, self.func(w_star, x), color='black', label='Ground Truth')
        plt.scatter(self.x_val, self.y_pred_val, color='red', label='Validation Set Model')
        plt.scatter(self.x_tr, self.y_pred_tr, color='blue', label='Training Set Model')
        plt.title('Polynomial Regression Results')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()

        #Plot Training set model
        plt.subplot(2,3,5)
        plt.scatter(self.x_tr, self.y_tr, color='orange', label='Training Set')
        plt.scatter(self.x_tr, self.y_pred_tr, color='blue', label='Training Set Model')
        plt.title('Polynomial Regression Model for Training Set')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()

        #Plot validation set model
        plt.subplot(2,3,6)
        plt.scatter(self.x_val, self.y_val, color='green', label='Validation Set')
        plt.scatter(self.x_val, self.y_pred_val, color='red', label='Validation Set Model')
        plt.title('Polynomial Regression Model for Validation Set')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()

        plt.savefig(f'{name}.png')


#####################################################################################
###########################            Function           ###########################
#####################################################################################

#Define pointwise polynomial feature matrix
#   x : input vector
#   i : index for samples
#   j : index for feature
def polynomialFeatures(x, i, j):
    return x[i]**j

#Define pointwise polynomial feature matrix
#   w_star : 
#   x : linspace on domain
def getPolynomial(w_star, x):
    result = w_star[0]
    for i in range(1, len(w_star)):
        result += w_star[i] * np.power(x,i)

    return result

#####################################################################################
###########################            Main           ###############################
#####################################################################################

def main():
    #Display results from noisy dataset
    dataset = NoisyDataset(polynomialFeatures, getPolynomial)
    dataset.get()

    plt.figure(figsize=(10,5))
    dataset.saveAndDisplayDataset(plt)

    #test regression (DIVERGES when alpha > .001)
    model = RegressionModel(dataset, getPolynomial)
    model.train(run=True)

    #plot and save results
    plt.figure(figsize=(15,10))
    model.saveAndDisplayResults(plt, name='number7_results')

    #Test with small training set
    dataset2 = NoisyDataset(polynomialFeatures, getPolynomial)
    dataset2.get(sizes=np.array([10,500]))

    plt.figure(figsize=(10,5))
    dataset2.saveAndDisplayDataset(plt, name='low_scatter')

    model2 = RegressionModel(dataset2, getPolynomial)
    model2.train(run=True)

    plt.figure(figsize=(15,10))
    model2.saveAndDisplayResults(plt, name='smalltrain_new')

if __name__ == "__main__":
    main()
