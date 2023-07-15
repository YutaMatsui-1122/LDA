import numpy as np
from pre_processing import pre_processing
import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument('--dir-name', type=str,  help='name of directory for saving model')
args = parser.parse_args()

def Latent_Dirichlet_Allocation(w, model_dir, K=10):
    #initialize
    

    I = 1000
    D, N = w.shape
    V = np.max(w)+1
    alpha = np.repeat(1, K)
    beta = np.repeat(1, V)
    theta = np.random.dirichlet(alpha=alpha, size = D) # D×K
    phi = np.random.dirichlet(alpha=beta, size = K) # K×V

    z = np.array([[np.random.randint(K) if w[d][n]!=V else K for n in range(N)] for d in range(D)])

    # Gibbs sampling
    for i in range(I):
        # Sampling theta 
        alpha_hat = np.sum(np.identity(K+1)[z], axis=1)[:,:-1] + alpha
        for d in range(D):
            theta[d] = np.random.dirichlet(alpha_hat[d])

        # Sampling z
        for d in range(D):
            theta_d = theta[d,:]
            for n in range(N):
                if w[d][n] == -1:
                    z[d][n] = -1 
                else:
                    eta = phi[:, w[d][n]]  * theta_d
                    eta = eta / np.sum(eta)
                    z[d][n] = np.random.choice(K ,p= eta)

        # Sampling phi
        for k in range(K):
            index = np.where(z==k)
            p = np.bincount(w[index[0],index[1]],minlength=V) + beta
            phi[k] = np.random.dirichlet(p)
        
        print(i)
        np.save(os.path.join(model_dir,"theta"),theta)
        np.save(os.path.join(model_dir,"phi"),phi)
        np.save(os.path.join(model_dir,"z"),z)

w, index2word_dict = pre_processing()
model_dir = os.path.join("model",args.dir_name)
os.makedirs(model_dir,exist_ok=True)
np.save(os.path.join(model_dir,"w"),w)
np.save(os.path.join(model_dir,"index2word_dict"),index2word_dict,allow_pickle=True)
Latent_Dirichlet_Allocation(w, model_dir, K=10)
