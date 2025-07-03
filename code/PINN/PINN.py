import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the neural network
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 6)  # Outputs for y_hif, y_o2, y_p300, y_p53, y_casp, y_kp
        )

    def forward(self, t):
        return self.net(t)

# Parameters
params = {
    'a_hif': 1.52, 'a_o2': 1.8, 'a_p53': 0.05, 'a3': 0.9,
    'a4': 0.2, 'a5': 0.001, 'a7': 0.7, 'a8': 0.06,
    'a9': 0.1, 'a10': 0.7, 'a11': 0.2, 'a12': 0.1,
    'a13': 0.1, 'a14': 0.05
}

# Initial conditions
y0 = torch.tensor([[1., 0., 0., 0., 0., 0.]], device=device)
t0 = torch.tensor([[0.]], requires_grad=True, device=device)

# Collocation points (0 to 100)
t_train = torch.linspace(0, 100, 200).view(-1, 1).to(device)
t_train.requires_grad = True

# Instantiate model
model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
mse_loss = nn.MSELoss()

# Derivative helper function
def gradients(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]

# Training loop
for epoch in range(5000):
    optimizer.zero_grad()

    y_pred = model(t_train)
    y_hif, y_o2, y_p300, y_p53, y_casp, y_kp = y_pred[:,0:1], y_pred[:,1:2], y_pred[:,2:3], y_pred[:,3:4], y_pred[:,4:5], y_pred[:,5:6]

    dy_hif = gradients(y_hif, t_train)
    dy_o2 = gradients(y_o2, t_train)
    dy_p300 = gradients(y_p300, t_train)
    dy_p53 = gradients(y_p53, t_train)
    dy_casp = gradients(y_casp, t_train)
    dy_kp = gradients(y_kp, t_train)

    # ODE residuals
    f1 = params['a_hif'] - params['a3']*y_o2*y_hif - params['a4']*y_hif*y_p300 - params['a7']*y_p53*y_hif - dy_hif
    f2 = params['a_o2'] - params['a3']*y_o2*y_hif + params['a4']*y_hif*y_p300 - params['a11']*y_o2 - dy_o2
    f3 = -params['a4']*y_hif*y_p300 - params['a5']*y_p300*y_p53 + params['a8'] - dy_p300
    f4 = params['a_p53'] - params['a5']*y_p300*y_p53 - params['a9']*y_p53 - dy_p53
    f5 = params['a9']*y_p53 + params['a12'] - params['a13']*y_casp - dy_casp
    f6 = -params['a10']*y_casp*y_kp + params['a11']*y_o2 - params['a14']*y_kp - dy_kp

    loss_f = (f1**2 + f2**2 + f3**2 + f4**2 + f5**2 + f6**2).mean()
    y0_pred = model(t0)
    loss_ic = mse_loss(y0_pred, y0)
    loss = loss_f + loss_ic

    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.5f}")

# Generate test data
t_test = torch.linspace(0, 100, 200).view(-1, 1).to(device)
with torch.no_grad():
    y_test = model(t_test).cpu().numpy()
t_test = t_test.cpu().numpy()

# Labels for each variable
labels = ['y_hif', 'y_o2', 'y_p300', 'y_p53', 'y_casp', 'y_kp']

# Plot each graph separately
for i in range(6):
    plt.figure(figsize=(10, 5))
    plt.plot(t_test, y_test[:, i], 'b-', linewidth=2, label=labels[i])
    
    # Set axis properties
    plt.xticks(range(0, 101, 20))  # x-axis: 0 to 100 in steps of 20
    plt.yticks([x * 0.2 for x in range(16)])  # y-axis: 0 to 3 in steps of 0.2
    plt.xlim(0, 100)
    plt.ylim(0, 3)
    
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title(f'Time Evolution of {labels[i]}')
    plt.grid(True)
    plt.legend()
    plt.show()