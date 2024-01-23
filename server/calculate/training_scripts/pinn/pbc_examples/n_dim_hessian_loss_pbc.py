import sys
sys.path.append('../')

import copy
import time
import argparse
import matplotlib
import numpy as np
import argparse
import torch
import torch.nn
import tqdm

from net_pbc import *
from systems_pbc import *
from utils import *
from visualize import *
from PyHessian.pyhessian import hessian_pinn

matplotlib.rcParams['figure.figsize'] = [18, 12]

def get_params(model_orig,  model_perb, direction, alpha):
    for m_orig, m_perb, d in zip(model_orig.parameters(), model_perb.parameters(), direction):
        m_perb.data = m_orig.data + alpha * d
    return model_perb

# device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FLAG = True if torch.cuda.is_available() else False

parser = argparse.ArgumentParser(description='High Dim Hessian of PINNs')
parser.add_argument('--system', type=str, default='convection', help='System to study.')
parser.add_argument('--seed', type=int, default=0, help='Random initialization.')
parser.add_argument('--N_f', type=int, default=100, help='Number of collocation points to sample.')
parser.add_argument('--optimizer_name', type=str, default='LBFGS', help='Optimizer of choice.')
parser.add_argument('--lr', type=float, default=1.0, help='Learning rate.')
parser.add_argument('--L', type=float, default=1.0, help='Multiplier on loss f.')

parser.add_argument('--xgrid', type=int, default=256, help='Number of points in the xgrid.')
parser.add_argument('--nt', type=int, default=100, help='Number of points in the tgrid.')
parser.add_argument('--nu', type=float, default=1.0, help='nu value that scales the d^2u/dx^2 term. 0 if only doing advection.')
parser.add_argument('--rho', type=float, default=1.0, help='reaction coefficient for u*(1-u) term.')
parser.add_argument('--beta', type=float, default=1.0, help='beta value that scales the du/dx term. 0 if only doing diffusion.')
parser.add_argument('--u0_str', default='sin(x)', help='str argument for initial condition if no forcing term.')
parser.add_argument('--source', default=0, type=float, help="If there's a source term, define it here. For now, just constant force terms.")

parser.add_argument('--layers', type=str, default='50,50,50,50,1', help='Dimensions/layers of the NN, minus the first layer.')
parser.add_argument('--net', type=str, default='DNN', help='The net architecture that is to be used.')
parser.add_argument('--activation', default='tanh', help='Activation to use in the network.')
parser.add_argument('--loss_style', default='mean', help='Loss for the network (MSE, vs. summing).')

parser.add_argument('--visualize', default=False, help='Visualize the solution.')
parser.add_argument('--save_model', default=False, help='Save the model for analysis later.')

parser.add_argument('--dim', default=3, help='dimension for hessian loss values calculation')
parser.add_argument('--steps', default=40, help='steps for hessian loss values calculation')
parser.add_argument('--points', default=64000, help='total points while sampling for hessian loss values calculation')

args = parser.parse_args()

nu = args.nu
beta = args.beta
rho = args.rho

START = -2.0
END = 2.0
STEPS = int(args.steps)
DIM = int(args.dim)
POINTS = int(args.points)

if args.system == 'diffusion': # just diffusion
    beta = 0.0
    rho = 0.0
elif args.system == 'convection':
    nu = 0.0
    rho = 0.0
elif args.system == 'rd': # reaction-diffusion
    beta = 0.0
elif args.system == 'reaction':
    nu = 0.0
    beta = 0.0

print('nu', nu, 'beta', beta, 'rho', rho)

# parse the layers list here
orig_layers = args.layers
layers = [int(item) for item in args.layers.split(',')]

############################
# Process data
############################

x = np.linspace(0, 2*np.pi, args.xgrid, endpoint=False).reshape(-1, 1) # not inclusive
t = np.linspace(0, 1, args.nt).reshape(-1, 1)
X, T = np.meshgrid(x, t) # all the X grid points T times, all the T grid points X times
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None])) # all the x,t "test" data

# remove initial and boundaty data from X_star
t_noinitial = t[1:]
# remove boundary at x=0
x_noboundary = x[1:]
X_noboundary, T_noinitial = np.meshgrid(x_noboundary, t_noinitial)
X_star_noinitial_noboundary = np.hstack((X_noboundary.flatten()[:, None], T_noinitial.flatten()[:, None]))

# sample collocation points only from the interior (where the PDE is enforced)
X_f_train = sample_random(X_star_noinitial_noboundary, args.N_f)

if 'convection' in args.system or 'diffusion' in args.system:
    u_vals = convection_diffusion(args.u0_str, nu, beta, args.source, args.xgrid, args.nt)
    G = np.full(X_f_train.shape[0], float(args.source))
elif 'rd' in args.system:
    u_vals = reaction_diffusion_discrete_solution(args.u0_str, nu, rho, args.xgrid, args.nt)
    G = np.full(X_f_train.shape[0], float(args.source))
elif 'reaction' in args.system:
    u_vals = reaction_solution(args.u0_str, rho, args.xgrid, args.nt)
    G = np.full(X_f_train.shape[0], float(args.source))
else:
    print("WARNING: System is not specified.")

u_star = u_vals.reshape(-1, 1) # Exact solution reshaped into (n, 1)
Exact = u_star.reshape(len(t), len(x)) # Exact on the (x,t) grid

xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T)) # initial condition, from x = [-end, +end] and t=0
uu1 = Exact[0:1,:].T # u(x, t) at t=0
bc_lb = np.hstack((X[:,0:1], T[:,0:1])) # boundary condition at x = 0, and t = [0, 1]
uu2 = Exact[:,0:1] # u(-end, t)

# generate the other BC, now at x=2pi
t = np.linspace(0, 1, args.nt).reshape(-1, 1)
x_bc_ub = np.array([2*np.pi]*t.shape[0]).reshape(-1, 1)
bc_ub = np.hstack((x_bc_ub, t))

u_train = uu1 # just the initial condition
X_u_train = xx1 # (x,t) for initial condition

layers.insert(0, X_u_train.shape[-1])

############################
# Load the model
############################

set_seed(args.seed) # for weight initialization

# model = PhysicsInformedNN_pbc(args.system, X_u_train, u_train, X_f_train, bc_lb, bc_ub, layers, G, nu, beta, rho,
#                             args.optimizer_name, args.lr, args.net, args.L, args.activation, args.loss_style)
# model.train()
model = torch.load(f"saved_models/pretrained_{args.system}_u0{args.u0_str}_nu{nu}_beta{beta}_rho{rho}_Nf{args.N_f}_{args.layers}_L{args.L}_source{args.source}_seed{args.seed}.pt", map_location=device)
model.dnn.eval()

# make sure the numbers of sampling points is less than the total number of points
FULL_CUBE = False
if POINTS >= STEPS ** DIM:
    POINTS = STEPS ** DIM
    FULL_CUBE = True
    print("Full cube calculation. Total points: ", POINTS)
else:
    print("Random sampling cube calculation. Total points: ", POINTS)

############################
# Calculate the Hessian
############################

start = time.time()

# load loss criterion
criterion = torch.nn.CrossEntropyLoss()

# make the model a copy
model_init = copy.deepcopy(model.dnn)
model_init.eval()
model_perb = copy.deepcopy(model.dnn)
model_perb.eval()
model_current = copy.deepcopy(model.dnn)
model_current.eval()

# data that the evaluator will use when evaluating loss
x, y = iter(X_f_train).__next__()

# send the model and datato GPU if available
if FLAG == True:
    model.cuda()
    model_perb.cuda()
    model_current.cuda()
    x = x.cuda()
    y = y.cuda()

# Generate the loss values array using BFS
# Create a coordinate sampling array for loss values
loss_coordinates_list = []
if FULL_CUBE == False:
    pbar = tqdm.tqdm(total=POINTS, desc="Generating sampling coordinates in the subspace")
    while len(loss_coordinates_list) < POINTS:
        t = ()
        for j in range(DIM):
            t += (np.random.randint(0, STEPS),)
        if t not in loss_coordinates_list:
            loss_coordinates_list.append(t)
            pbar.update(1)
    pbar.close()
    loss_coordinates = np.asarray(loss_coordinates_list)
else:
    print("Generating full cube coordinates in the subspace")
    loss_coordinates = np.empty([POINTS, DIM], dtype=int)
    for i in range(POINTS):
        loss_coordinates[i] = np.unravel_index(i, (STEPS,)*DIM)
        
# verify the shape of the loss coordinates
print(loss_coordinates.shape)

# Create a data matrix to store loss values
data_matrix = np.empty([POINTS, 1], dtype=float)
# Fill array with initial value (e.g., -1)
data_matrix.fill(-1)
# print(data_matrix)
print(data_matrix.shape)

# calculate the hessian eigenvalues and eigenvectors
x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)

hessian_comp = hessian_pinn(model, model_init, criterion, data=(x, t), cuda=FLAG)
top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=DIM)
print("Top eigenvalues: ", top_eigenvalues)

lams = np.linspace(START, END, STEPS).astype(np.float32)

# calculate the hessian loss values
for j in tqdm.tqdm(range(POINTS), desc="Calculating sampling loss values in the subspace"):
    # adjust the model and fill with a loss with corresponding model parameters
    next_pos = tuple(loss_coordinates[j])
    model_current = copy.deepcopy(model_init)
    for i in range(DIM):
        model_perb = get_params(model_current, model_perb, top_eigenvector[i], lams[next_pos[i]])
        model_current = copy.deepcopy(model_perb)
    # calculate the loss value
    # outputs = model_current(torch.cat([x,t], dim=1))
    # data_matrix[j] = criterion(outputs, t).detach().cpu().numpy()
    model.dnn = copy.deepcopy(model_current)
    if torch.is_grad_enabled():
        model.optimizer.zero_grad()
    u_pred = model_current(torch.cat([x, t], dim=1))
    u_pred_lb = model.net_u(model.x_bc_lb, model.t_bc_lb)
    u_pred_ub = model.net_u(model.x_bc_ub, model.t_bc_ub)
    if model.nu != 0:
        u_pred_lb_x, u_pred_ub_x = model.net_b_derivatives(u_pred_lb, u_pred_ub, model.x_bc_lb, model.x_bc_ub)
    f_pred = model.net_f(model.x_f, model.t_f)
    
    if model.loss_style == 'mean':
        loss_u = torch.mean((t - u_pred) ** 2)
        loss_b = torch.mean((u_pred_lb - u_pred_ub) ** 2)
        if model.nu != 0:
            loss_b += torch.mean((u_pred_lb_x - u_pred_ub_x) ** 2)
        loss_f = torch.mean(f_pred ** 2)
    elif model.loss_style == 'sum':
        loss_u = torch.mean((t - u_pred) ** 2)
        loss_b = torch.sum((u_pred_lb - u_pred_ub) ** 2)
        if model.nu != 0:
            loss_b += torch.sum((u_pred_lb_x - u_pred_ub_x) ** 2)
        loss_f = torch.sum(f_pred ** 2)

    loss = loss_u + loss_b + model.L*loss_f
    data_matrix[j] = loss.detach().cpu().numpy()

# save the loss values
np.save(f"high_dim_hessian/pretrained_{args.system}_u0{args.u0_str}_nu{nu}_beta{beta}_rho{rho}_Nf{args.N_f}_{args.layers}_L{args.L}_source{args.source}_seed{args.seed}_dim{DIM}_points{POINTS}_hessian.npy", data_matrix)
# save the coordinates
np.save(f"high_dim_hessian/pretrained_{args.system}_u0{args.u0_str}_nu{nu}_beta{beta}_rho{rho}_Nf{args.N_f}_{args.layers}_L{args.L}_source{args.source}_seed{args.seed}_dim{DIM}_points{POINTS}_hessian_coordinates.npy", loss_coordinates)

end = time.time()
print('Time taken: ', end - start)