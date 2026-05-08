import torch
import torch.nn as nn
import torch.optim as optim

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Simple tensor operation test
print("\n--- Tensor operations ---")
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
print(f"a + b = {a + b}")

# GPU computation test
if torch.cuda.is_available():
    a_gpu = a.to("cuda")
    b_gpu = b.to("cuda")
    c_gpu = a_gpu + b_gpu
    print(f"GPU result: {c_gpu.cpu()}")

# Simple neural network test
print("\n--- NN test: y = 2x + noise ---")
X = torch.linspace(-5, 5, 100).reshape(-1, 1).to("cuda" if torch.cuda.is_available() else "cpu")
y = 2.0 * X.squeeze() + torch.randn(100).to(X.device) * 0.5

model = nn.Sequential(nn.Linear(1, 4), nn.ReLU(), nn.Linear(4, 1)).to(X.device)
opt = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(200):
    opt.zero_grad()
    pred = model(X).squeeze()
    loss = loss_fn(pred, y)
    loss.backward()
    opt.step()

print(f"Final loss: {loss.item():.6f}")

with torch.no_grad():
    test_input = torch.tensor([[3.0]], device=X.device)
    prediction = model(test_input)
    print(f"f(3.0) ≈ {prediction.item():.4f} (expected ~6.0)")

print("\nAll tests passed!")
