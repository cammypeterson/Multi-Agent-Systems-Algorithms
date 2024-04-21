import numpy as np
import params as P
import matplotlib.pyplot as plt
from pyoptsparse import Optimization, OPT
import jax



def average_data_service_at_xy(x,y,uav_index):
    #this function finds the average data service at a given location x,y from the given UAV position
    #we wish to maximize the average data service
    lambda_x = P.B/(P.N*P.alpha[uav_index])

def elevation_angle(x,y,uav_index):
    #this function finds the elevation angle of the given UAV at a given location x,y
    h = P.h_UAV[uav_index]
    d = np.sqrt((x-P.x_locations[uav_index])**2 + (y-P.y_locations[uav_index])**2 + h**2)
    theta = np.arcsin(h/d)
    return theta
    

def recieved_power(x,y,uav_index):
    #Equation 3 from the the paper
    P_i = P.P[uav_index]
    K_o = ((4*np.pi*P.f_c)/(3e9))**2
    theta_i = elevation_angle(x,y,uav_index)
    di_squared = (x-P.x_locations[uav_index])**2 + (y-P.y_locations[uav_index])**2 + P.h_UAV[uav_index]**2
    if(180/np.pi * theta_i - 15 < 0):
        P_los = 0
    else:
        P_los = P.b1*(180/np.pi * theta_i - 15)**P.b2 #equation 2
        # print(x,y,P.h_UAV[uav_index])
        # print("Error: theta_i is less than 15 degrees")
        # print(theta_i*180/np.pi)
    # P_los = .5
    # P_los = P.b1*(180/np.pi * theta_i)**P.b2
    P_nlos = 1 - P_los
    
    den = K_o*di_squared*(P_los * P.mu_los+P_nlos*P.mu_nlos)
    num = P_i/(P.N/P.num_UAVs)
    return num/den

def gamma(x,y,uav_index):
    #equation 4 from the paper
    return recieved_power(x,y,uav_index)/(recieved_interference(x,y,uav_index) + 10**(P.N_O/10)/1000)
    
def recieved_interference(x,y,uav_index):
    interference = 0
    for j in range(P.num_UAVs):
        if j != uav_index:
            interference += recieved_power(x,y,j)
    return P.beta_interference_factor*interference

def effective_data_tranmission_time(uav_index):
    return P.hover_time[uav_index] - 0.01*(P.N/P.num_UAVs)**2

def lambda_i(uav_index):
    sum = 0
    for i in range(P.num_UAVs):
        sum += effective_data_tranmission_time(i)
    print(sum)
    return P.B/(P.N*P.alpha[uav_index])*sum



lambda_i_precomputed = np.zeros(P.num_UAVs)
for i in range(P.num_UAVs):
    lambda_i_precomputed[i] = lambda_i(i)
print(lambda_i_precomputed)

def J(x,y,uav_index):
    return -lambda_i_precomputed[uav_index]*np.log2(1+gamma(x,y,uav_index))

def omega_i(uav_index):
    sum = 0
    for i in range(P.num_UAVs):
        sum += P.alpha[i] * effective_data_tranmission_time(i) 
    return P.alpha[uav_index]*effective_data_tranmission_time(uav_index)/sum


        
precomputed_omega_i = np.zeros(P.num_UAVs)
for i in range(P.num_UAVs):
    precomputed_omega_i[i] = omega_i(i)


def integrate_d_i(psi):
    #d_i is defined in equation 27
    int_vals = np.zeros(P.num_UAVs)
    for x in P.x_int:
        for y in P.y_int:
            index = -1
            cost_val = np.inf
            for i in range(P.num_UAVs):
                if J(x,y,i) - psi[i] < cost_val:
                    index = i
                    cost_val = J(x,y,i) - psi[i]
            int_vals[index] += P.f(x,y)*P.dx*P.dy
    return int_vals
                


def compute_grad_f(psi):
    #defined in equation 27
    grad = np.zeros(P.num_UAVs)
    integral_over_d_i = integrate_d_i(psi)
    for i in range(P.num_UAVs):
        grad[i] = precomputed_omega_i[i] - integral_over_d_i[i] 
    return grad

    
    
    
def find_partitions_from_psi(psi):
    #Line 25 from Algorithm 1
    #finds the partitions from the given psi
    cell_index = np.zeros((len(P.x_int),len(P.y_int)))
    
    for i,x in enumerate(P.x_int):
        for j,y in enumerate(P.y_int):
            min_val = np.inf
            min_index = -1
            for k in range(P.num_UAVs):
                if (J(x,y,k) - psi[k]) < min_val:
                    min_val = J(x,y,k) - psi[k]
                    min_index = k
            cell_index[i,j] = min_index
                
    return cell_index

def plot_cell_partitions(cell_index):
    #plots the cell partitions
    [X_plot,Y_plot] = np.meshgrid(P.x_int,P.y_int)
    
    plt.figure()
    # c = plt.contourf(X_plot,Y_plot,cell_index,levels = np.arange(0,P.num_UAVs+1)-1)
    c = plt.pcolormesh(X_plot,Y_plot,cell_index)
    plt.colorbar(c)
    for i in range(P.num_UAVs):
        plt.plot(P.x_locations[i],P.y_locations[i],'ro')
    plt.show()


def find_max_cost(uav_index):
    max_cost = -np.inf
    for x in P.x_int:
        for y in P.y_int:
            max_cost = max(max_cost,J(x,y,uav_index))
    return max_cost


def integrate_c_transform_over_d(psi):
    int_val = 0
    for x in P.x_int:
        for y in P.y_int:
            max_cost = -np.inf
            for k in range(P.num_UAVs):
                max_cost = max(max_cost,J(x,y,k)-psi[k])
            int_val += max_cost*P.f(x,y)*P.dx*P.dy
    return int_val
            

def objefctive_function(psi):
    #defined in equation 22
    sum = 0
    for i in range(P.num_UAVs):
        sum += omega_i(i) * psi[i]
    return sum + integrate_c_transform_over_d(psi)

    
def find_optimal_partiions():
    # psi = np.random.uniform(1,100, P.num_UAVs)
    psi = np.ones(P.num_UAVs)*100

    # 
    grad_f = compute_grad_f(psi)
    iter = 0
    # while np.linalg.norm(grad_f) > 1e-3:
    for i in range(100):
        print(i)
        grad_f = compute_grad_f(psi)
        print(np.linalg.norm(grad_f))
        k=100
        e = 1
        psi_plus = psi + e*grad_f
        obj_func_psi = objefctive_function(psi)
        print(obj_func_psi)
        if objefctive_function(psi_plus) < objefctive_function(psi):
            while obj_func_psi < objefctive_function(psi_plus):
                k = k+1
                e = 2**(k-1)*e
                psi_plus = psi + e*grad_f
        else:
            while obj_func_psi >= objefctive_function(psi_plus):
                k = k+1
                e = 2**(-k+1)*e
                psi_plus = psi + e*grad_f
        psi = psi_plus
        iter += 1
                
        
    
    return psi



def get_gradient_finite_diff(f,x,h):
    #calculate gradient using finite differencing
    x = np.array(x)

    #store the function value at x
    fx = f(x)

    #initialize the jacobian matrix (# functions by # variables
    grad = np.zeros((len(fx),len(x)))

    #this loops through each column of the Jacobian
    for i in range(len(x)):
        #the next three lines creates a step vector where all elements are zeros except for the current step direction
        epsilon = np.zeros(len(x))
        step = h * (1 + abs(x[i]))
        epsilon[i] = step

        #add the step to the x vector
        xi = x + epsilon

        grad[:,i] = (f(xi) - fx)/step
        # print("grad[:,i]",grad[:,i])

    return grad

    
def optimize_with_IPOPT():
                
    def objfunc(xdict):
        psi = xdict['psi']
        # obj = -next_measurement_covariance_determinant(pos, estimatedParams_list, estimatedRadarCovariance_list)
        obj = -objefctive_function(psi)
        funcs = {}
        funcs['obj'] = obj
        fail = False
        return funcs, fail

    optProb = Optimization("find best measurement location", objfunc)
    optProb.addVarGroup(name = "psi", nVars = P.num_UAVs, varType = 'c', value = np.ones(P.num_UAVs))
    optProb.addObj("obj")
    opt = OPT("ipopt")

    opt.options['hsllib'] = '/home/grant/packages/ThirdParty-HSL/.libs/libcoinhsl.so'
    opt.options['linear_solver'] = 'ma97'
    opt.options['print_level'] = 5
        
    opt.options['max_iter'] = 10
    opt.options['tol'] = 1e-8
    sol = opt(optProb, sens = 'FD')
    return sol.xStar['psi']

if __name__ == '__main__':
    psi = optimize_with_IPOPT()
    # psi = np.ones(P.num_UAVs)*10
    # psi = np.random.uniform(1,100, P.num_UAVs)
    # print("finite diff", get_gradient_finite_diff(objefctive_function,psi,1e-6))
    # print("grad", compute_grad_f(psi))
    # psi = find_optimal_partiions()
    print(psi)
    cell_index = find_partitions_from_psi(psi)
    plot_cell_partitions(cell_index)
    