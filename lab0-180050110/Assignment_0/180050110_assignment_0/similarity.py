import argparse
import time
import numpy as np 
import matplotlib.pyplot as plt

def d(x,y):
    '''
    Given x and y where each is an np arrays of size (dim,1), compute L2 distance between them
    '''
    
    ## ADD CODE HERE ##
    return np.linalg.norm(x - y)

def pairwise_similarity_looped(X):
    '''
    Given X, Y where each is an np array of size (num_points_1,dim) and (num_points_2,dim), 
    return K, an array having size (num_points_1,num_points_2) according to the problem given
    '''
    ## STEP 1 - Initialize K as a numpy array - ADD CODE TO COMPUTE n1, n2 ##
    
    n = X.shape[0]
    K = np.zeros((n,n))#.astype('float64')

    ## STEP 2 - Loop and compute  -- COMPLETE THE LOOP BELOW ##

    for i in range(n):
        for j in range (n):
            K[i][j] = d(X[i,:],X[j,:])

    return K 


def pairwise_similarity_vec(X):
    '''
    Given X, Y where each is an np array of size (num_points_1,dim) and (num_points_2,dim), 
    return K, an array having size (num_points_1,num_points_2) according to the problem given

    This problem can be simplified in the following way - 
    Each entry in K has three terms (as seen in problem 2.1 (a)).
    Hence, first  compute the norm for all points, reshape it suitably,
    then compute the dot product.
    All of these can be done by using on the *, np.matmul, np.sum(), and transpose operators.
    '''


    ## ADD CODE TO COMPUTE K ##
    n = X.shape[0]
    tmp = np.ones((n,n)) * np.sum(np.square(X), axis = 1)
    return np.sqrt(np.maximum(0,tmp + tmp.T - 2 * np.matmul(X,X.T)))

def plot_graph():
    fig, axs = plt.subplots(2, 2) 

    dim = 20
    x = []
    y_loop = []
    y_vec = []
    num_max = 200
    for num in range(1,num_max):
        X = np.random.normal(0.,1.,size=(num,dim))    
        t1 = time.time()
        K_loop = pairwise_similarity_looped(X)
        t2 = time.time()
        K_vec  = pairwise_similarity_vec(X)
        t3 = time.time()
        # if not np.allclose(K_loop,K_vec):
        #     print(K_loop)
        #     print(K_vec)
        #     print("num =",num)
        #     assert(False)
        x.append(num)
        y_loop.append(t2-t1)
        y_vec.append(t3-t2)
    axs[0, 0].plot(x, y_loop)
    axs[0, 0].set(ylabel = 'wrt num')
    axs[0, 0].set_title('loop')
    axs[0, 1].plot(x, y_vec)
    axs[0, 1].set_title('vec')

    num = 50
    x = []
    y_loop = []
    y_vec = []
    dim_max = 1000
    for dim in range(1,dim_max):
        X = np.random.normal(0.,1.,size=(num,dim))    
        t1 = time.time()
        K_loop = pairwise_similarity_looped(X)
        t2 = time.time()
        K_vec  = pairwise_similarity_vec(X)
        t3 = time.time()
        # if not np.allclose(K_loop,K_vec):
        #     print(K_loop)
        #     print(K_vec)
        #     print("num =",num)
        #     assert(False)
        x.append(dim)
        y_loop.append(t2-t1)
        y_vec.append(t3-t2)
    axs[1, 0].plot(x, y_loop)
    axs[1, 0].set(ylabel = 'wrt dim')
    axs[1, 1].plot(x, y_vec)

    axs[0,0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axs[1,0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axs[0,1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axs[1,1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.show()

if __name__ == '__main__':

    # uncomment below line for graphs
    plot_graph()

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--num', type=int, default=5,
                    help='Number of samples to generate')
    parser.add_argument('--seed', type=int, default=42,
                    help='Seed for random generator')
    parser.add_argument('--dim', type=float, default=2,
                    help='Lambda parameter for the distribution')

    args = parser.parse_args()

    np.random.seed(args.seed)

    X = np.random.normal(0.,1.,size=(args.num,args.dim))
    # Y = np.random.normal(1.,1.,size=(args.num,args.dim))

    t1 = time.time()
    K_loop = pairwise_similarity_looped(X)
    t2 = time.time()
    K_vec  = pairwise_similarity_vec(X)
    t3 = time.time()

    assert np.allclose(K_loop,K_vec)   # Checking that the two computations match

    np.savetxt("problem_2_loop.txt",K_loop)
    np.savetxt("problem_2_vec.txt",K_vec)
    print("Vectorized time : {}, Loop time : {}".format(t3-t2,t2-t1))






            