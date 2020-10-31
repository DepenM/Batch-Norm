import os
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import Dataset
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

def plot_Hessian(model, loss, trainloader, num_eigenthings, step, grad=None):
    start_time = timeit.default_timer()
    eigvals_Hess, eigvecs_Hess = compute_hessian_eigenthings(model, trainloader,
                                                     loss, num_eigenthings, mode='lanczos', full_dataset=True,
                                                     type='Hessian', grad=grad,
                                                     max_steps = num_eigenthings*100, tol=1e-2, max_samples=10000)
    elapsed = timeit.default_timer() - start_time
    print(elapsed)
    for j in range(len(eigvals_Hess)):
        eigenvals_Hess[j].append(eigvals_Hess[num_eigenthings-j-1])

    Hess_eigvecs = np.array(eigvecs_Hess).transpose()

    Hess_eigenvecs.append(Hess_eigvecs)

    if step in step_gap*np.array(steps_to_obs):
        Hess_eigenvecs_steps.append(Hess_eigvecs)

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
                                               shuffle=False, num_workers=0)
trainloader2 = torch.utils.data.DataLoader(trainset_1, batch_size=len(trainset_1),
                                               shuffle=False, num_workers=0)
test_length = 10000
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)
indices = np.random.choice(len(testset), test_length, replace=False)
testset_1 = torch.utils.data.Subset(testset, indices)
testloader = torch.utils.data.DataLoader(testset_1, batch_size=len(testset_1),
                                         shuffle=False, num_workers=0)

top_eigens_observe = k
BN_momentum = 0.1
num_layers = 3
neurons = 100
monitor_all = False
#Whether the numerator in normalization is mean-shifted or not
mean_shift = True
p_norm = 2
#whether the normalizing moment is centered or nor
centered = False
is_bias = True
gamma_1= False
beta_1 = False
Hess = True
act = 'relu'
lr = 0.1
tot_epochs = 200
change_epoch = 20
best_val_acc = 0
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
if gamma_1 and beta_1:
    gammas = []
    abs_gammas = []
    betas = []
    abs_betas = []
    abs_gammas_norms = []
elif gamma_1:
    gammas = []
    abs_gammas = []
    abs_gammas_norms = []
elif beta_1:
    betas = []
    abs_betas = []

for i in range(num_layers):
    w_norm_layers.append([])
    pre_acts.append([])
    pre_acts_grad.append([])
    cond_numbers.append([])
    sgrad_norm_layers.append([])
    max_sv.append([])
    min_sv.append([])
    if i!=num_layers-1:
        acts.append([])
        acts_grad.append([])
    if act!='linear' and i!=num_layers-1:
        fracs.append([])
    if i!=num_layers-1:
        if gamma_1 and beta_1:
            gammas.append([])
            abs_gammas.append([])
            betas.append([])
            abs_betas.append([])
            abs_gammas_norms.append([])
            gamma_grad_norm.append([])
            norm_gamma.append([])
        elif gamma_1:
            gammas.append([])
            abs_gammas.append([])
            abs_gammas_norms.append([])
            gamma_grad_norm.append([])
            norm_gamma.append([])
        elif beta_1:
            betas.append([])
            abs_betas.append([])

    if i != num_layers-1:
        grad_ratios.append([])


step_gap = 100
if Hess:
    steps_to_obs = [0,10,20,40,60,80,100]
    #steps_to_obs = [0,1,2,3,4,6]
    grad_succ_angles = []
    sgrad_succ_angles = []
    Hess_eigenvecs_steps = []
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
        if gamma_1:
            self.gamma = nn.ParameterList()
        if beta_1:
            self.beta = nn.ParameterList()
        self.running_mean = []
        self.running_norm = []
        for i in range(n_layers):
            if i==0:
                if n_layers==1:
                    self.linears.append(nn.Linear(d, k, bias=is_bias))
                else:
                    self.linears.append(nn.Linear(d, neurons, bias=is_bias))
                    if mean_shift:
                        self.running_mean.append(torch.zeros(1,neurons).to(device))
                    self.running_norm.append(torch.zeros(1,neurons).to(device))
                    if gamma_1 and beta_1:
                        self.gamma.append(nn.Parameter(torch.ones(1, neurons).to(device)))
                        self.beta.append(nn.Parameter(torch.zeros(1, neurons).to(device)))
                    elif gamma_1:
                        self.gamma.append(nn.Parameter(torch.ones(1, neurons).to(device)))
                    elif beta_1:
                        self.beta.append(nn.Parameter(torch.zeros(1, neurons).to(device)))
            elif i==n_layers-1:
                self.linears.append(nn.Linear(neurons, k, bias=is_bias))
            else:
                self.linears.append(nn.Linear(neurons, neurons, bias=is_bias))
                if mean_shift:
                    self.running_mean.append(torch.zeros(1, neurons).to(device))
                self.running_norm.append(torch.zeros(1, neurons).to(device))
                if gamma_1 and beta_1:
                    self.gamma.append(nn.Parameter(torch.ones(1, neurons).to(device)))
                    self.beta.append(nn.Parameter(torch.zeros(1, neurons).to(device)))
                elif gamma_1:
                    self.gamma.append(nn.Parameter(torch.ones(1, neurons).to(device)))
                elif beta_1:
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

    def forward(self, x, step = 1, record = 0, update_mean_norm = 0):
        x = x.view(-1,self.d)
        for i, l in enumerate(self.linears):
            x = l(x)
            mean = torch.mean(x, dim=0, keepdim=True)
            if centered:
                norm1 = torch.pow((torch.norm(torch.abs(x - torch.mean(x, dim=0, keepdim=True)), p=p_norm, dim=0, keepdim=True)**p_norm)/x.shape[0] + 1e-5, 1/p_norm)
            else:
                norm1 = torch.pow((torch.norm(x, p=p_norm, dim=0, keepdim=True)**p_norm)/x.shape[0] + 1e-5, 1/p_norm)
            if i!=self.layers-1:
                if self.training:
                    if mean_shift:
                        x = x-mean
                    x = x/norm1
                    if update_mean_norm:
                        if mean_shift:
                            self.running_mean[i] = BN_momentum * mean.detach() + (1 - BN_momentum) * \
                                                   self.running_mean[i].detach()
                        self.running_norm[i] = BN_momentum * norm1.detach() + (1 - BN_momentum) * \
                                                  self.running_norm[i].detach()
                else:
                    if mean_shift:
                        x = x - self.running_mean[i]
                    x = x/(self.running_norm[i])
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

                    if gamma_1:
                        gammas[i].append(torch.sum(self.gamma[i]).detach().cpu() / self.gamma[i].shape[1])
                        abs_gammas[i].append(
                            torch.sum(torch.abs(self.gamma[i])).detach().cpu() / self.gamma[i].shape[1])
                        abs_gammas_norms[i].append(
                            torch.sum(torch.abs(self.gamma[i]) / norm1).detach().cpu() /
                            self.gamma[i].shape[1])
                        norm_gamma[i].append(torch.norm(self.gamma[i]).detach().cpu())
                    if beta_1:
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
    best_val_acc_1 = 0
    check = False
    best_epoch_past = 0
    for epoch in range(tot_epochs):  # loop over the dataset multiple times
        net = net.train()
        running_loss = 0.0
        acc = 0.0
        for i, data in enumerate(trainloader, 0):
            if i==0 and epoch==0:
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

                plot_Hessian(net, criterion, trainloader2, top_eigens_observe, step, grad_fin)
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
                    tot_grad += torch.norm(param.grad) ** 2
                for j in range(num_layers):
                    if name == 'linears.' + str(j) + '.weight':
                        grads_temp[j] = torch.norm(param.grad)
                        sgrad_norm_layers[j].append(torch.norm(param.grad).detach().cpu())
                        w_norm_layers[j].append(torch.norm(param).detach().cpu())
                    if name == 'gamma.' + str(j) and param.grad is not None:
                        gamma_grad_norm[j].append(torch.norm(param.grad).detach().cpu())
            sgrad_norm.append(torch.sqrt(tot_grad))
            if monitor_all:
                for j in range(num_layers - 1):
                    grad_ratios[j].append(grads_temp[num_layers - 1] / grads_temp[j])

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
            lr_changes_steps.append((epoch+1)*train_steps)
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
    plt.xlabel('Epochs')
    plt.ylabel('Norm')
    plt.title('Distance moved')
    plt.show()

    A = np.zeros((len(params),len(params)))
    for i in range(len(params)):
        for j in range(len(params)):
            A[i,j] = torch.norm(params[i] - params[j])

    sns.heatmap(A)
    plt.xlabel('Steps')
    plt.ylabel('Steps')
    plt.title('Difference in norm between the parameters at the two steps')
    plt.show()

    grad_angle = []
    sg_angle = []
    for j in range(len(params)-1):
        grad_angle.append(torch.dot(params[-1]-params[j],-grads[j])/(torch.norm(params[-1]-params[j])*torch.norm(grads[j])))
        sg_angle.append(torch.dot(params[-1] - params[j], -s_grads[j]) / (
                    torch.norm(params[-1] - params[j]) * torch.norm(s_grads[j])))

    plt.plot(np.arange(len(grad_angle))*step_gap, grad_angle, label='grad_angle')
    plt.plot(np.arange(len(sg_angle))*step_gap, sg_angle, label='sg_angle')
    plt.legend()
    plt.title('Gradient/Stochastic Gradient angle with residual')
    plt.show()

    plt.plot(np.arange(len(grad_succ_angles)), grad_succ_angles, label='grad_succ_angles')
    plt.legend()
    plt.title('Gradient successive angles')
    plt.show()

    plt.plot(np.arange(len(sgrad_succ_angles)), sgrad_succ_angles, label='sgrad_succ_angles')
    plt.legend()
    plt.title('Stochastic gradient successive angles')
    plt.close()

    for i in range(top_eigens_observe):
        plt.plot(np.arange(len(eigenvals_Hess[i]))*step_gap, eigenvals_Hess[i], label=str(i))
    plt.legend()
    plt.title('Eigenvals_Hessian')
    plt.show()

    plt.plot(np.arange(len(grad_norm))*step_gap, grad_norm, label='norm')
    plt.legend()
    plt.title('Gradient norm')
    plt.show()

print('Best_val_acc for lr = is ' + str(best_val_acc))

plt.plot(list(range(len(train_loss))), train_loss, label='train_loss')
plt.plot(list(range(len(train_loss))), test_loss, label='test_loss')
plt.legend(loc='upper right')
plt.show()

plt.plot(list(range(len(train_loss))), train_acc, label='train_acc')
plt.plot(list(range(len(train_loss))), test_acc, label='test_acc')
plt.legend(loc='upper right')
plt.show()
