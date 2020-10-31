import os
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import Dataset
from hessian import Hessian
from torchvision import transforms
from hessian_eigenthings import compute_hessian_eigenthings
import timeit
import seaborn as sns
import random

plt.style.use('seaborn')
seed = 3689
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(seed)

def plot_Hessian(model, loss, trainloader, num_eigenthings, dir, step, grad=None):
    start_time = timeit.default_timer()
    eigvals_Hess, eigvecs_Hess = compute_hessian_eigenthings(model, trainloader,
                                                     loss, num_eigenthings, mode='lanczos', full_dataset=True,
                                                     type='Hessian', grad=grad,
                                                     max_steps = num_eigenthings*100, tol=1e-2, max_samples=10000)
    eigvals_G, eigvecs_G = compute_hessian_eigenthings(model, trainloader,
                                                             loss, num_eigenthings, mode='lanczos', full_dataset=True,
                                                             type='G', grad=grad,
                                                             max_steps=num_eigenthings * 100, tol=1e-2, max_samples=10000)
    eigvals_H, eigvecs_H = compute_hessian_eigenthings(model, trainloader,
                                                             loss, num_eigenthings, mode='lanczos', full_dataset=True,
                                                             type='H', grad=grad,
                                                             max_steps=num_eigenthings * 100, tol=1e-2, max_samples=10000)
    eigvals_covar, eigvecs_covar = compute_hessian_eigenthings(model, trainloader,
                                                             loss, num_eigenthings, mode='lanczos', full_dataset=True,
                                                             type='covar', grad=grad,
                                                             max_steps=num_eigenthings * 100, tol=1e-2, max_samples=10000)
    eigvals_covar_res, eigvecs_covar_res = compute_hessian_eigenthings(model, trainloader,
                                                             loss, num_eigenthings, mode='lanczos', full_dataset=True,
                                                             type='covar_residual', grad=grad,
                                                             max_steps=num_eigenthings * 100, tol=1e-2, max_samples=10000)

    elapsed = timeit.default_timer() - start_time
    print(elapsed)
    for j in range(len(eigvals_Hess)):
        eigenvals_Hess[j].append(eigvals_Hess[num_eigenthings-j-1])
        eigenvals_G[j].append(eigvals_G[num_eigenthings-j-1])
        eigenvals_H[j].append(eigvals_H[num_eigenthings-j-1])
        eigenvals_covar[j].append(eigvals_covar[num_eigenthings-j-1])
        eigenvals_covar_res[j].append(eigvals_covar_res[num_eigenthings-j-1])

    Hess_eigvecs = np.array(eigvecs_Hess).transpose()
    G_eigvecs = np.array(eigvecs_G)
    H_eigvecs = np.array(eigvecs_H)
    covar_eigvecs = np.array(eigvecs_covar)
    covar_res_eigvecs = np.array(eigvecs_covar_res)

    Hess_eigenvecs.append(Hess_eigvecs)
    covar_eigenvecs.append(covar_eigvecs.transpose())

    if step in step_gap*np.array(steps_to_obs):
        Hess_eigenvecs_steps.append(Hess_eigvecs)

    for j in range(num_eigenthings):
        G_Hess_proj[j].append((np.linalg.norm(
            np.matmul(G_eigvecs[-(j+1):,:].transpose(), np.matmul(G_eigvecs[-(j+1):,:], Hess_eigvecs[:,-(j+1):])))**2))
        H_Hess_proj[j].append((np.linalg.norm(
            np.matmul(H_eigvecs[-(j+1):,:].transpose(), np.matmul(H_eigvecs[-(j+1):,:], Hess_eigvecs[:,-(j+1):])))**2))
        covar_Hess_proj[j].append((np.linalg.norm(
            np.matmul(covar_eigvecs[-(j+1):,:].transpose(), np.matmul(covar_eigvecs[-(j+1):,:], Hess_eigvecs[:,-(j+1):])))**2))
        covar_res_Hess_proj[j].append((np.linalg.norm(
            np.matmul(covar_res_eigvecs[-(j+1):,:].transpose(), np.matmul(covar_res_eigvecs[-(j+1):,:], Hess_eigvecs[:,-(j+1):])))**2))
        for k1 in range(len(Hess_eigenvecs_steps)):
            Hess_hess_proj[k1*num_eigenthings + j].append((np.linalg.norm(
                np.matmul(Hess_eigvecs[:, -(j + 1):],
                          np.matmul(Hess_eigvecs[:, -(j + 1):].transpose(), Hess_eigenvecs_steps[k1][:, -(j + 1):])))**2))

    start_time = timeit.default_timer()
    Hess = Hessian(loader=trainloader,
                   model=model,
                   hessian_type='Hessian',
                   grad=grad,
                   )

    Hess_eigval, \
    Hess_eigval_density = Hess.LanczosApproxSpec(init_poly_deg=64,
                                                 # iterations used to compute spectrum range
                                                 poly_deg=256)  # the higher the parameter the better the approximation

    G = Hessian(loader=trainloader,
                   model=model,
                   hessian_type='G',
                   grad=grad,
                   )

    G_eigval, \
    G_eigval_density = G.LanczosApproxSpec(init_poly_deg=64,
                                                 # iterations used to compute spectrum range
                                                 poly_deg=256)  # the higher the parameter the better the approximation

    H = Hessian(loader=trainloader,
                   model=model,
                   hessian_type='H',
                   grad=grad,
                   )

    H_eigval, \
    H_eigval_density = H.LanczosApproxSpec(init_poly_deg=64,
                                                 # iterations used to compute spectrum range
                                                 poly_deg=256)  # the higher the parameter the better the approximation

    covar = Hessian(loader=trainloader,
                   model=model,
                   hessian_type='covar',
                   grad=grad,
                   )

    covar_eigval, \
    covar_eigval_density = covar.LanczosApproxSpec(init_poly_deg=64,
                                                 # iterations used to compute spectrum range
                                                 poly_deg=256)  # the higher the parameter the better the approximation

    covar_res = Hessian(loader=trainloader,
                   model=model,
                   hessian_type='covar_residual',
                   grad=grad,
                   )

    covar_res_eigval, \
    covar_res_eigval_density = covar_res.LanczosApproxSpec(init_poly_deg=64,
                                                 # iterations used to compute spectrum range
                                                 poly_deg=256)  # the higher the parameter the better the approximation

    plt.figure()
    plt.semilogy(Hess_eigval, Hess_eigval_density, label='Hessian')
    plt.semilogy(G_eigval, G_eigval_density, label='G')
    plt.semilogy(H_eigval, H_eigval_density, label='H')
    plt.semilogy(eigvals_Hess, np.ones(num_eigenthings) / 1000, '*', label='Hess_vals')
    plt.semilogy(eigvals_G, np.ones(num_eigenthings) / 10000, 'o', label='G_vals')
    plt.semilogy(eigvals_H, np.ones(num_eigenthings) / 100000, 'v', label='H_vals')
    plt.legend(loc = 'upper right')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Density of Spectrum')
    plt.savefig(dir + '/GH/step' + str(step) + '.png')
    plt.close()

    plt.figure()
    plt.semilogy(Hess_eigval, Hess_eigval_density, label='Hessian')
    plt.semilogy(covar_eigval, covar_eigval_density, label='covar')
    plt.semilogy(covar_res_eigval, covar_res_eigval_density, label='covar_res')
    plt.semilogy(eigvals_Hess, np.ones(num_eigenthings) / 1000, '*', label='Hess_vals')
    plt.semilogy(eigvals_covar, np.ones(num_eigenthings) / 100000, 'o', label='covar_vals')
    plt.semilogy(eigvals_covar_res, np.ones(num_eigenthings) / 100000, 'v', label='covar_res_vals')
    plt.legend(loc = 'upper right')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Density of Spectrum')
    plt.savefig(dir + '/covar/step' + str(step) + '.png')
    plt.close()
    elapsed = timeit.default_timer() - start_time
    print(elapsed)

    if grad is not None:
        proj = 0
        n = torch.norm(grad) ** 2
        for i in range(Hess_eigvecs.shape[1]):
            t = torch.tensor(Hess_eigvecs[:,-(i+1)]).to(device)
            proj += torch.dot(t, grad) ** 2
            grad_proj[i].append(proj/n)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#generating training and testing data
d=784
k=10

transform = transforms.Compose(
    [  # transforms.Resize(10),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

train_length = 10000
batch_size = 64
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)
indices = np.random.choice(len(trainset), train_length, replace=False)
trainset_1 = torch.utils.data.Subset(trainset, indices)
trainloader = torch.utils.data.DataLoader(trainset_1, batch_size=batch_size,
                                               shuffle=True, num_workers=0)
trainloader2 = torch.utils.data.DataLoader(trainset_1, batch_size=len(trainset_1),
                                               shuffle=True, num_workers=4)
test_length = 10000
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)
indices = np.random.choice(len(testset), test_length, replace=False)
testset_1 = torch.utils.data.Subset(testset, indices)
testloader = torch.utils.data.DataLoader(testset_1, batch_size=len(testset_1),
                                         shuffle=False, num_workers=0)

top_eigens_observe = k
BN_momentum = 0.1
num_layers = 2
neurons = 100
is_bias = True
monitor_all = True
BN=True
gamma_1= False
beta_1 = False
Hess = True
act = 'relu'
lr = 0.01
tot_epochs = 2
change_epoch = 20

dir = './Hessian_analysis'
if BN:
    dir = dir + '/BN'
else:
    dir = dir + '/no-BN'

CHECK_FOLDER = os.path.isdir(dir)

# If folder doesn't exist, then create it.
if not CHECK_FOLDER:
    os.makedirs(dir)
    print("created folder : ", dir)

if Hess:
    dir2 = dir + '/GH'
    dir3 = dir + '/covar'

    CHECK_FOLDER = os.path.isdir(dir2)

    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER:
        os.makedirs(dir2)
        print("created folder : ", dir2)

    CHECK_FOLDER = os.path.isdir(dir3)

    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER:
        os.makedirs(dir3)
        print("created folder : ", dir3)

for i in range(num_layers):
    dir_new = dir + '/layer' + str(i+1)
    CHECK_FOLDER = os.path.isdir(dir_new)

    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER:
        os.makedirs(dir_new)
        print("created folder : ", dir_new)

sgrad_norm_layers = []
sgrad_norm = []
gamma_grad_norm = []
lr_changes_steps = []
lr_changes_epochs = []
if act!='linear':
    fracs = []
cond_numbers = []
max_sv = []
min_sv = []
grad_ratios = []
w_norm_layers = []
pre_acts = []
acts = []
norm_gamma = []
acts_grad = []
pre_acts_grad = []
if BN:
    if gamma_1 and beta_1:
        gammas = []
        abs_gammas = []
        betas = []
        abs_betas = []
        abs_gammas_sigmas = []
    elif gamma_1:
        gammas = []
        abs_gammas = []
        abs_gammas_sigmas = []
    elif beta_1:
        betas = []
        abs_betas = []

for i in range(num_layers):
    cond_numbers.append([])
    w_norm_layers.append([])
    sgrad_norm_layers.append([])
    max_sv.append([])
    min_sv.append([])
    pre_acts.append([])
    pre_acts_grad.append([])
    if i!=num_layers-1:
        acts.append([])
        acts_grad.append([])
    if act!='linear' and i!=num_layers-1:
        fracs.append([])
    if BN and i!=num_layers-1:
        if gamma_1 and beta_1:
            gammas.append([])
            abs_gammas.append([])
            betas.append([])
            abs_betas.append([])
            abs_gammas_sigmas.append([])
            gamma_grad_norm.append([])
            norm_gamma.append([])
        elif gamma_1:
            gammas.append([])
            abs_gammas.append([])
            abs_gammas_sigmas.append([])
            gamma_grad_norm.append([])
            norm_gamma.append([])
        elif beta_1:
            betas.append([])
            abs_betas.append([])

    if i != num_layers-1:
        grad_ratios.append([])

step_gap = 100
if Hess:
    #steps_to_obs = [0,10,20,40,60,80,100]
    steps_to_obs = [0,1,2,3,4,6]
    grad_succ_angles = []
    sgrad_succ_angles = []
    Hess_eigenvecs_steps = []
    Hess_hess_proj = []
    grads = []
    s_grads = []
    params = []
    Hess_eigenvecs = []
    covar_eigenvecs = []
    grad_proj_proxy = []
    grad_norm = []
    grad_proj = []
    G_Hess_proj = []
    H_Hess_proj = []
    covar_Hess_proj = []
    covar_res_Hess_proj = []
    eigenvals_Hess = []
    eigenvals_G = []
    eigenvals_H = []
    eigenvals_covar = []
    eigenvals_covar_res = []
    for i in range(top_eigens_observe):
        for j in range(len(steps_to_obs)):
            Hess_hess_proj.append([])
        eigenvals_Hess.append([])
        eigenvals_G.append([])
        eigenvals_H.append([])
        eigenvals_covar.append([])
        eigenvals_covar_res.append([])
        grad_proj.append([])
        G_Hess_proj.append([])
        H_Hess_proj.append([])
        covar_Hess_proj.append([])
        covar_res_Hess_proj.append([])


class TestNet(nn.Module):
    def __init__(self, n_layers, d, neurons, k, is_bias, act):
        super(TestNet, self).__init__()
        self.linears = nn.ModuleList()
        self.d = d
        if BN:
            if gamma_1:
                self.gamma = nn.ParameterList()
            if beta_1:
                self.beta = nn.ParameterList()
        self.running_mean = []
        self.running_var = []
        for i in range(n_layers):
            if i==0:
                if n_layers==1:
                    self.linears.append(nn.Linear(d, k, bias=is_bias))
                else:
                    self.linears.append(nn.Linear(d, neurons, bias=is_bias))
                    self.running_mean.append(torch.zeros(1,neurons).to(device))
                    self.running_var.append(torch.zeros(1,neurons).to(device))
                    if BN:
                        if gamma_1:
                            self.gamma.append(nn.Parameter(torch.ones(1, neurons).to(device)))
                        if beta_1:
                            self.beta.append(nn.Parameter(torch.zeros(1, neurons).to(device)))
            elif i==n_layers-1:
                self.linears.append(nn.Linear(neurons, k, bias=is_bias))
            else:
                self.linears.append(nn.Linear(neurons, neurons, bias=is_bias))
                self.running_mean.append(torch.zeros(1, neurons).to(device))
                self.running_var.append(torch.zeros(1, neurons).to(device))
                if BN:
                    if gamma_1:
                        self.gamma.append(nn.Parameter(torch.ones(1, neurons).to(device)))
                    if beta_1:
                        self.beta.append(nn.Parameter(torch.zeros(1, neurons).to(device)))
        self.layers = n_layers
        self.act = act

    def save_pre_acts_grad(self, i):
        def hook(grad):
            pre_acts_grad[i].append(torch.norm(grad).detach().cpu())

        return hook

    def save_acts_grad(self, i):
        def hook(grad):
            acts_grad[i].append(torch.norm(grad).detach().cpu())

        return hook

    def forward(self, x, step = 1, record = 0, update_mean_var = 0):
        x = x.view(-1,self.d)
        for i, l in enumerate(self.linears):
            x = l(x)
            mean = torch.mean(x, dim=0, keepdim=True)
            var = torch.var(x, dim=0, keepdim=True)
            if BN:
                if i!=self.layers-1:
                    if self.training:
                        x = (x-mean)/(torch.sqrt(var + 1e-5))
                        if update_mean_var:
                            self.running_mean[i] = BN_momentum*mean + (1-BN_momentum)*self.running_mean[i]
                            self.running_var[i] = BN_momentum*var + (1-BN_momentum)*self.running_var[i]
                    else:
                        x = (x-self.running_mean[i])/(torch.sqrt(self.running_var[i] + 1e-5))
                    if gamma_1 and beta_1:
                        x = self.gamma[i]*x + self.beta[i]
                    elif gamma_1:
                        x = self.gamma[i] * x
                    elif beta_1:
                        x = x + self.beta[i]

            if record:
                if i!=self.layers-1:
                    if act=='relu':
                        t = x > 0
                        fracs[i].append((torch.sum(t).detach().cpu() + 0.)/(x.shape[0]*x.shape[1]))
                    elif act == 'hardtanh':
                        t = x < 1
                        t1 = x > -1
                        t2 = t*t1
                        fracs[i].append((torch.sum(t2).detach().cpu() + 0.) / (x.shape[0] * x.shape[1]))

                    if BN:
                        if gamma_1 and beta_1:
                            betas[i].append(torch.sum(self.beta[i]).detach().cpu()/(self.beta[i].shape[1]))
                            abs_betas[i].append(
                                torch.sum(torch.abs(self.beta[i]) / self.beta[i].shape[1]).detach().cpu())
                            gammas[i].append(torch.sum(self.gamma[i]).detach().cpu() / self.gamma[i].shape[1])
                            abs_gammas[i].append(
                                torch.sum(torch.abs(self.gamma[i])).detach().cpu() / self.gamma[i].shape[1])
                            norm_gamma[i].append(torch.norm(self.gamma[i]).detach().cpu())
                            abs_gammas_sigmas[i].append(
                                torch.sum(torch.abs(self.gamma[i]) / torch.sqrt(var + 1e-5)).detach().cpu() /
                                self.gamma[i].shape[1])
                        elif gamma_1:
                            gammas[i].append(torch.sum(self.gamma[i]).detach().cpu() / self.gamma[i].shape[1])
                            abs_gammas[i].append(
                                torch.sum(torch.abs(self.gamma[i])).detach().cpu() / self.gamma[i].shape[1])
                            abs_gammas_sigmas[i].append(
                                torch.sum(torch.abs(self.gamma[i]) / torch.sqrt(var + 1e-5)).detach().cpu() /
                                self.gamma[i].shape[1])
                            norm_gamma[i].append(torch.norm(self.gamma[i]).detach().cpu())
                        elif beta_1:
                            betas[i].append(torch.sum(self.beta[i]).detach().cpu() / (self.beta[i].shape[1]))
                            abs_betas[i].append(
                                torch.sum(torch.abs(self.beta[i]) / self.beta[i].shape[1]).detach().cpu())

                pre_acts[i].append(torch.norm(x).detach().cpu())
                x.register_hook(self.save_pre_acts_grad(i))

                if monitor_all:
                    matrix = np.array(self.linears[i].weight.cpu().data)
                    u = np.linalg.svd(matrix, compute_uv=False)
                    cond_numbers[i].append(np.max(u) / np.min(u))
                    max_sv[i].append(np.max(u))
                    min_sv[i].append(np.min(u))
                    if step%10 == 0:
                        plt.hist(u, bins=20)
                        plt.savefig(dir + '/layer' + str(i+1) + '/step' + str(step) + '.png')
                        plt.close()

            if i!=self.layers-1 and self.act!='linear':
                if self.act=='relu':
                    x = F.relu(x)
                elif self.act=='hardtanh':
                    x = F.hardtanh(x)

            if self.act!='linear' and i!=self.layers-1 and record:
                acts[i].append(torch.norm(x).detach().cpu())
                x.register_hook(self.save_acts_grad(i))
        return x

if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    if train_length%batch_size==0:
        train_steps = int(train_length/batch_size)
    else:
        train_steps = int(train_length/batch_size) + 1
    net = TestNet(num_layers, d, neurons, k, is_bias, act)
    net.to(device)
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    old_sgrad = None
    old_grad = None

    criterion = torch.nn.functional.cross_entropy
    curr_index = 0
    optimizer = optim.SGD(net.parameters(), lr)
    best_val_acc = 0
    check = False
    best_epoch_past = 0
    for epoch in range(tot_epochs):  # loop over the dataset multiple times
        net = net.train()
        running_loss = 0.0
        acc = 0.0
        for i, data in enumerate(trainloader, 0):
            if i==0:
                for name, param in net.named_parameters():
                    for j in range(num_layers):
                        if name == 'linears.' + str(j) + '.weight':
                            w_norm_layers[j].append(torch.norm(param).detach().cpu())

            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            step = epoch * train_steps + i
            outputs = net(inputs, step, 1, 1)
            loss = criterion(outputs, labels)

            if Hess:
                s_grad = torch.autograd.grad(loss, net.parameters(), retain_graph=True)
                for data2 in trainloader2:
                    inputs2, labels2 = data2[0].to(device), data2[1].to(device)
                    outputs2 = net(inputs2)
                    loss2 = criterion(outputs2, labels2)
                if step%step_gap==0:
                    grad = torch.autograd.grad(loss2, net.parameters(), create_graph=True)
                else:
                    grad = torch.autograd.grad(loss2, net.parameters())
                if old_sgrad is None:
                    old_sgrad = torch.cat([g.detach().cpu().contiguous().view(-1) for g in s_grad])
                else:
                    new_sgrad = torch.cat([g.detach().cpu().contiguous().view(-1) for g in s_grad])
                    sgrad_succ_angles.append(torch.dot(old_sgrad, new_sgrad)/(torch.norm(old_sgrad)*torch.norm(new_sgrad)))
                    old_sgrad = new_sgrad

                if old_grad is None:
                    old_grad = torch.cat([g.detach().cpu().contiguous().view(-1) for g in grad])
                else:
                    new_grad = torch.cat([g.detach().cpu().contiguous().view(-1) for g in grad])
                    grad_succ_angles.append(
                        torch.dot(old_grad, new_grad) / (torch.norm(old_grad) * torch.norm(new_grad)))
                    old_grad = new_grad

            if step%step_gap == 0 and Hess:
                params.append(torch.cat([g.detach().cpu().contiguous().view(-1) for g in net.parameters()]))
                s_grads.append(torch.cat([g.detach().cpu().contiguous().view(-1) for g in s_grad]))
                grads.append(torch.cat([g.detach().cpu().contiguous().view(-1) for g in grad]))
                Hg = torch.autograd.grad(grad, net.parameters(), grad)
                grad_fin = torch.cat([g.detach().contiguous().view(-1) for g in grad])
                Hg_fin = torch.cat([g.detach().contiguous().view(-1) for g in Hg])
                grad_norm.append(torch.norm(grad_fin))
                grad_proj_proxy.append(
                    torch.dot(grad_fin, Hg_fin) / (torch.norm(grad_fin) * torch.norm(Hg_fin)))

                plot_Hessian(net, criterion, trainloader2, top_eigens_observe, dir, step, grad_fin)
                del grad
                del s_grad
                del Hg
                del grad_fin
                del Hg_fin
                del loss2
                del inputs2
                del outputs2
            elif Hess:
                del grad
                del s_grad
                del loss2
                del inputs2
                del outputs2

            loss.backward()
            optimizer.step()

            indices = torch.argmax(outputs, 1)
            acc = acc + torch.sum(indices == labels)

            tot_grad = 0
            grads_temp = np.zeros((num_layers))
            for name, param in net.named_parameters():
                if param.grad is not None:
                    tot_grad += torch.norm(param.grad)**2
                for j in range(num_layers):
                    if name == 'linears.' + str(j) + '.weight':
                        grads_temp[j] = torch.norm(param.grad)
                        sgrad_norm_layers[j].append(torch.norm(param.grad).detach().cpu())
                        w_norm_layers[j].append(torch.norm(param).detach().cpu())
                    if name == 'gamma.' + str(j) and param.grad is not None:
                        gamma_grad_norm[j].append(torch.norm(param.grad).detach().cpu())
            sgrad_norm.append(torch.sqrt(tot_grad))
            if monitor_all:
                for j in range(num_layers-1):
                    grad_ratios[j].append(grads_temp[num_layers-1] / grads_temp[j])

            running_loss += loss.item()
            if i % 500 == 499:
                t = 1
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / i))

        train_loss.append(running_loss / (i + 1))
        train_acc.append(acc.item() / train_length)
        print('[%d] Train loss: %.3f, Train acc: %.3f' %
              (epoch + 1, running_loss / (i + 1), acc.item() / train_length))
        net = net.eval()
        acc = 0.0
        total_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = net(inputs)

                loss = criterion(outputs, labels)
                total_loss += loss.item()
                indices = torch.argmax(outputs, 1)
                acc = acc + torch.sum(indices == labels)

        print('[%d] Validation loss: %.3f, Validation acc: %.3f' %
              (epoch + 1, total_loss / (i + 1), acc.item() / test_length))

        test_loss.append(total_loss / (i + 1))
        test_acc.append(acc.item() / test_length)
        if acc.item() / test_length > best_val_acc:
            best_val_acc = acc.item() / test_length
            best_epoch_past = 0
        else:
            best_epoch_past += 1

        if best_epoch_past == change_epoch:
            lr = lr * 0.1
            if lr < 1e-3:
                break
            lr_changes_epochs.append(epoch)
            lr_changes_steps.append(step)
            print("At epoch " + str(epoch) + ", lr changed to " + str(lr))
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] * 0.1
            best_epoch_past = 0

if Hess:
    dist_moved = []
    for j in range(len(steps_to_obs)):
        dist_moved.append([])

    curr_index = 0
    for j in range(len(params)):
        if curr_index<len(steps_to_obs) and j==steps_to_obs[curr_index]:
            curr_index = curr_index + 1
        for k1 in range(curr_index):
            dist_moved[k1].append(torch.norm(params[steps_to_obs[k1]] - params[j]))

    for i in range(len(steps_to_obs)):
        plt.plot(steps_to_obs[i]*step_gap + step_gap*np.arange(len(dist_moved[i])), dist_moved[i], label=str(steps_to_obs[i]))
    plt.legend()
    plt.savefig(dir + '/dist_moved.png')
    plt.close()

    A = np.zeros((len(params),len(params)))
    for i in range(len(params)):
        for j in range(len(params)):
            A[i,j] = torch.norm(params[i] - params[j])

    sns.heatmap(A)
    plt.savefig(dir + '/dist_moved_heatmap.png')
    plt.close()

    grad_angle = []
    sg_angle = []
    for j in range(len(params)-1):
        grad_angle.append(torch.dot(params[-1]-params[j],-grads[j])/(torch.norm(params[-1]-params[j])*torch.norm(grads[j])))
        sg_angle.append(torch.dot(params[-1] - params[j], -s_grads[j]) / (
                    torch.norm(params[-1] - params[j]) * torch.norm(s_grads[j])))

    plt.plot(np.arange(len(grad_angle))*step_gap, grad_angle, label='grad_angle')
    plt.plot(np.arange(len(sg_angle))*step_gap, sg_angle, label='sg_angle')
    plt.legend()
    plt.savefig(dir + '/grad_angle_with_residual.png')
    plt.close()

    plt.plot(np.arange(len(grad_succ_angles)), grad_succ_angles, label='grad_succ_angles')
    plt.legend()
    plt.savefig(dir + '/grad_succ_angles.png')
    plt.close()

    plt.plot(np.arange(len(sgrad_succ_angles)), sgrad_succ_angles, label='sgrad_succ_angles')
    plt.legend()
    plt.savefig(dir + '/stochastic_grad_succ_angles.png')
    plt.close()

    res_Hess_proj = []
    sg_Hess_proj = []
    for j in range(top_eigens_observe):
        res_Hess_proj.append([])
        sg_Hess_proj.append([])
    for j in range(len(Hess_eigenvecs)):
        proj = 0
        proj1 = 0
        n = torch.norm(params[-1] - params[j]) ** 2
        n1 = torch.norm(s_grads[j])**2
        for i in range(top_eigens_observe):
            t = torch.tensor(Hess_eigenvecs[j][:, -(i + 1)])
            proj += torch.dot(t, params[-1] - params[j]) ** 2
            res_Hess_proj[i].append(proj/n)
            proj1 += torch.dot(t, s_grads[j]) ** 2
            sg_Hess_proj[i].append(proj1/n1)
    for i in range(top_eigens_observe):
        plt.plot(np.arange(len(res_Hess_proj[i]))*step_gap, res_Hess_proj[i], label = str(i+1))
    plt.legend()
    plt.savefig(dir + '/residual_Hess_proj.png')
    plt.close()

    for i in range(top_eigens_observe):
        plt.plot(np.arange(len(sg_Hess_proj[i]))*step_gap, sg_Hess_proj[i], label = str(i+1))
    plt.legend()
    plt.savefig(dir + '/sg_Hess_proj.png')
    plt.close()

    res_covar_proj = []
    sg_covar_proj = []
    g_covar_proj = []
    for j in range(top_eigens_observe):
        res_covar_proj.append([])
        sg_covar_proj.append([])
        g_covar_proj.append([])
    for j in range(len(covar_eigenvecs)):
        proj = 0
        proj1 = 0
        proj2 = 0
        n = torch.norm(params[-1] - params[j]) ** 2
        n1 = torch.norm(s_grads[j]) ** 2
        n2 = torch.norm(grads[j]) ** 2
        for i in range(top_eigens_observe):
            t = torch.tensor(covar_eigenvecs[j][:, -(i + 1)])
            proj += torch.dot(t, params[-1] - params[j]) ** 2
            res_covar_proj[i].append(proj / n)
            proj1 += torch.dot(t, s_grads[j]) ** 2
            sg_covar_proj[i].append(proj1 / n1)
            proj2 += torch.dot(t, grads[j]) ** 2
            g_covar_proj[i].append(proj2 / n2)
    for i in range(top_eigens_observe):
        plt.plot(np.arange(len(res_covar_proj[i]))*step_gap, res_covar_proj[i], label=str(i + 1))
    plt.legend()
    plt.savefig(dir + '/residual_covar_proj.png')
    plt.close()

    for i in range(top_eigens_observe):
        plt.plot(np.arange(len(sg_covar_proj[i]))*step_gap, sg_covar_proj[i], label=str(i + 1))
    plt.legend()
    plt.savefig(dir + '/sg_covar_proj.png')
    plt.close()

    for i in range(top_eigens_observe):
        plt.plot(np.arange(len(g_covar_proj[i]))*step_gap, g_covar_proj[i], label=str(i + 1))
    plt.legend()
    plt.savefig(dir + '/grad_covar_proj.png')
    plt.close()

    plt.plot(np.arange(len(grad_proj_proxy))*step_gap, grad_proj_proxy, label='grad_proj_proxy')
    plt.legend()
    plt.savefig(dir + '/grad_proj_proxy.png')
    plt.close()

    for i in range(top_eigens_observe):
        plt.plot(np.arange(len(G_Hess_proj[i]))*step_gap, G_Hess_proj[i], label=str(i+1))
    plt.legend()
    plt.savefig(dir + '/G_Hess_proj.png')
    plt.close()

    for i in range(top_eigens_observe):
        plt.plot(np.arange(len(H_Hess_proj[i]))*step_gap, H_Hess_proj[i], label=str(i+1))
    plt.legend()
    plt.savefig(dir + '/H_Hess_proj.png')
    plt.close()

    for i in range(top_eigens_observe):
        plt.plot(np.arange(len(covar_Hess_proj[i]))*step_gap, covar_Hess_proj[i], label=str(i+1))
    plt.legend()
    plt.savefig(dir + '/covar_Hess_proj.png')
    plt.close()

    for i in range(top_eigens_observe):
        plt.plot(np.arange(len(covar_res_Hess_proj[i]))*step_gap, covar_res_Hess_proj[i], label=str(i+1))
    plt.legend()
    plt.savefig(dir + '/covar_res_Hess_proj.png')
    plt.close()

    for i in range(top_eigens_observe):
        plt.plot(np.arange(len(grad_proj[i]))*step_gap, grad_proj[i], label=str(i+1))
    plt.legend()
    plt.savefig(dir + '/grad_proj.png')
    plt.close()

    for i in range(len(steps_to_obs)):
        for j in range(top_eigens_observe):
            plt.plot(steps_to_obs[i]*step_gap + np.arange(len(Hess_hess_proj[i*top_eigens_observe + j]))*step_gap, Hess_hess_proj[i*top_eigens_observe + j], label = str(j+1))
        plt.legend()
        plt.title('Starting time t=' + str(step_gap*steps_to_obs[i]))
        plt.savefig(dir + '/Hess_hess_proj_' + str(step_gap*steps_to_obs[i]))
        plt.close()

    for i in range(top_eigens_observe):
        plt.plot(np.arange(len(eigenvals_Hess[i]))*step_gap, eigenvals_Hess[i], label=str(i))
    plt.legend()
    plt.savefig(dir + '/eigenvals_Hess.png')
    plt.close()

    for i in range(top_eigens_observe):
        plt.plot(np.arange(len(eigenvals_H[i]))*step_gap, eigenvals_H[i], label=str(i))
    plt.legend()
    plt.savefig(dir + '/eigenvals_H.png')
    plt.close()

    for i in range(top_eigens_observe):
        plt.plot(np.arange(len(eigenvals_G[i]))*step_gap, eigenvals_G[i], label=str(i))
    plt.legend()
    plt.savefig(dir + '/eigenvals_G.png')
    plt.close()

    for i in range(top_eigens_observe):
        plt.plot(np.arange(len(eigenvals_covar[i]))*step_gap, eigenvals_covar[i], label=str(i))
    plt.legend()
    plt.savefig(dir + '/eigenvals_covar.png')
    plt.close()

    for i in range(top_eigens_observe):
        plt.plot(np.arange(len(eigenvals_covar_res[i]))*step_gap, eigenvals_covar_res[i], label=str(i))
    plt.legend()
    plt.savefig(dir + '/eigenvals_covar_res.png')
    plt.close()

    plt.plot(np.arange(len(grad_norm))*step_gap, grad_norm, label='norm')
    plt.legend()
    plt.savefig(dir + '/norm.png')
    plt.close()

print('Best_val_acc is ' + str(best_val_acc))

plt.plot(list(range(len(train_loss))), train_loss, label='train_loss')
plt.plot(list(range(len(train_loss))), test_loss, label='test_loss')
plt.legend(loc='upper right')
plt.savefig(dir + '/loss.png')
plt.close()

plt.plot(list(range(len(train_loss))), train_acc, label='train_acc')
plt.plot(list(range(len(train_loss))), test_acc, label='test_acc')
plt.legend(loc='upper right')
plt.savefig(dir + '/acc.png')
plt.close()

print(lr_changes_epochs)
print(lr_changes_steps)

if monitor_all:
    a = len(lr_changes_steps)
    b = np.array(list(range(step+1)))
    for j in range(num_layers):
        temp = np.array(cond_numbers[j])
        for k in range(len(lr_changes_steps)):
            if k == 0:
                fin = np.ma.masked_where(b > lr_changes_steps[k], temp)
            else:
                fin = np.ma.masked_where((b < lr_changes_steps[k - 1]) | (b > lr_changes_steps[k]), temp)
            plt.plot(list(range(len(cond_numbers[j]))), fin)
        if len(lr_changes_steps)==0:
            fin = cond_numbers[j]
        else:
            fin = np.ma.masked_where(b < lr_changes_steps[k], temp)
        plt.plot(list(range(len(cond_numbers[j]))), fin)
        plt.savefig(dir + '/cond_numb_Weight_' + str(j+1) + '.png')
        plt.close()

    for j in range(num_layers):
        temp = np.array(max_sv[j])
        for k in range(len(lr_changes_steps)):
            if k == 0:
                fin = np.ma.masked_where(b > lr_changes_steps[k], temp)
            else:
                fin = np.ma.masked_where((b < lr_changes_steps[k - 1]) | (b > lr_changes_steps[k]), temp)
            plt.plot(list(range(len(max_sv[j]))), fin)
        if len(lr_changes_steps)==0:
            fin = max_sv[j]
        else:
            fin = np.ma.masked_where(b < lr_changes_steps[k], temp)
        plt.plot(list(range(len(max_sv[j]))), fin)
        plt.savefig(dir + '/max_sv_Weight_' + str(j+1) + '.png')
        plt.close()

    for j in range(num_layers):
        temp = np.array(min_sv[j])
        for k in range(len(lr_changes_steps)):
            if k == 0:
                fin = np.ma.masked_where(b > lr_changes_steps[k], temp)
            else:
                fin = np.ma.masked_where((b < lr_changes_steps[k - 1]) | (b > lr_changes_steps[k]), temp)
            plt.plot(list(range(len(cond_numbers[j]))), fin)
        if len(lr_changes_steps)==0:
            fin = min_sv[j]
        else:
            fin = np.ma.masked_where(b < lr_changes_steps[k], temp)
        plt.plot(list(range(len(min_sv[j]))), fin)
        plt.savefig(dir + '/min_sv_Weight_' + str(j+1) + '.png')
        plt.close()

if act!='linear':
    for j in range(num_layers-1):
        plt.plot(list(range(len(fracs[j]))), fracs[j], label='Layer_' + str(j))
    plt.legend(loc='upper right')
    plt.savefig(dir + '/act_neurons_frac.png')
    plt.close()

if BN:
    if gamma_1:
        for j in range(num_layers-1):
            plt.plot(list(range(len(gammas[j]))), gammas[j], label='Layer_' + str(j))
        plt.legend(loc='upper right')
        plt.savefig(dir + '/avg_gamma_val.png')
        plt.close()

        for j in range(num_layers-1):
            plt.plot(list(range(len(abs_gammas[j]))), abs_gammas[j], label='Layer_' + str(j))
        plt.legend(loc='upper right')
        plt.savefig(dir + '/avg_abs_gamma_val.png')
        plt.close()

        for j in range(num_layers-1):
            plt.plot(list(range(len(abs_gammas_sigmas[j]))), abs_gammas_sigmas[j], label='Layer_' + str(j))
            if j == 0:
                plt.savefig(dir + '/avg_abs_gamma_sigma_0.png')
                plt.close()
        plt.legend(loc='upper right')
        plt.savefig(dir + '/avg_abs_gamma_sigma.png')
        plt.close()

        for j in range(num_layers-1):
            plt.plot(list(range(len(abs_gammas_sigmas[j]))), 1 / np.array(abs_gammas_sigmas[j]),
                     label='Layer_' + str(j))
            if j == 0:
                plt.savefig(dir + '/avg_abs_sigma_gamma_0.png')
                plt.close()
        plt.legend(loc='upper right')
        plt.savefig(dir + '/avg_abs_sigma_gamma.png')
        plt.close()
    if beta_1:
        for j in range(num_layers-1):
            plt.plot(list(range(len(betas[j]))), betas[j], label='Layer_' + str(j))
        plt.legend(loc='upper right')
        plt.savefig(dir + '/avg_beta_val.png')
        plt.close()

        for j in range(num_layers-1):
            plt.plot(list(range(len(abs_betas[j]))), abs_betas[j], label='Layer_' + str(j))
        plt.legend(loc='upper right')
        plt.savefig(dir + '/avg_abs_beta_val.png')
        plt.close()

if monitor_all:
    a = len(lr_changes_steps)
    b = np.array(list(range(step+1)))
    for j in range(num_layers-1):
        temp = np.array(grad_ratios[j])
        for k in range(len(lr_changes_steps)):
            if k == 0:
                fin = np.ma.masked_where(b > lr_changes_steps[k], temp)
            else:
                fin = np.ma.masked_where((b < lr_changes_steps[k - 1]) | (b > lr_changes_steps[k]), temp)
            plt.plot(list(range(len(grad_ratios[j]))), fin)
        if len(lr_changes_steps)==0:
            fin = grad_ratios[j]
        else:
            fin = np.ma.masked_where(b < lr_changes_steps[k], temp)
        plt.plot(list(range(len(grad_ratios[j]))), fin)
        plt.savefig(dir + '/grad_ratio_lastby' + str(j+1) + '.png')
        plt.close()

        temp = 1 / np.array(grad_ratios[j])
        for k in range(len(lr_changes_steps)):
            if k == 0:
                fin = np.ma.masked_where(b > lr_changes_steps[k], temp)
            else:
                fin = np.ma.masked_where((b < lr_changes_steps[k - 1]) | (b > lr_changes_steps[k]), temp)
            plt.plot(list(range(len(grad_ratios[j]))), fin)
        if len(lr_changes_steps)==0:
            fin = 1/np.array(grad_ratios[j])
        else:
            fin = np.ma.masked_where(b < lr_changes_steps[k], temp)
        plt.plot(list(range(len(grad_ratios[j]))), fin)
        plt.savefig(dir + '/grad_ratio_' + str(j + 1) + 'bylast.png')
        plt.close()

for j in range(num_layers):
    plt.plot(np.arange(len(sgrad_norm_layers[j])), sgrad_norm_layers[j])
    plt.savefig(dir + '/sgrad_norm_layer_' + str(j+1) + '.png')
    plt.close()

plt.plot(np.arange(len(sgrad_norm)), sgrad_norm)
plt.savefig(dir + '/sgrad_norm.png')
plt.close()

