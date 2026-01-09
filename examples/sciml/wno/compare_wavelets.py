import os
import torch
import torch.nn as nn
import numpy as np
from timeit import default_timer
import matplotlib.pyplot as plt
import sys

# Ensure project src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src")))

from scirex.torch.sciml.models.wno.wno_model import WaveConv2d
from example_poisson import generate_poisson_2d_data, UnitGaussianNormalizer, WNO2d

def run_experiment(wavelet_name, n_epochs=50):
    print(f"\n>>> Running Experiment with Wavelet: {wavelet_name}")
    
    # Parameters
    ntrain, ntest = 1000, 100
    nx, ny = 64, 64
    batch_size = 20
    learning_rate = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data
    input_f, output_u = generate_poisson_2d_data(ntrain + ntest, nx, ny)
    x_normalizer = UnitGaussianNormalizer(input_f[:ntrain])
    y_normalizer = UnitGaussianNormalizer(output_u[:ntrain]).to(device)
    
    input_f_norm = x_normalizer.encode(input_f)
    output_u_norm = y_normalizer.encode(output_u.to(device)).cpu()
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(input_f_norm[:ntrain], output_u_norm[:ntrain]),
        batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(input_f_norm[ntrain:], output_u_norm[ntrain:]),
        batch_size=batch_size, shuffle=False
    )
    
    # Model - Update the internal WaveConv2d to use the target wavelet
    class CustomWNO2d(WNO2d):
        def __init__(self, in_channels, out_channels, width, size, level=3, n_layers=4, padding=2, wavelet='db4'):
            super().__init__(in_channels, out_channels, width, size, level, n_layers, padding)
            # Re-initialize the conv layers with the specific wavelet
            padded_size = [s + padding for s in size]
            self.conv_layers = nn.ModuleList([
                WaveConv2d(width, width, level, padded_size, wavelet=wavelet) 
                for _ in range(n_layers)
            ])

    model = CustomWNO2d(in_channels=3, out_channels=1, width=64, size=[nx, ny], level=3, n_layers=4, padding=2, wavelet=wavelet_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    criterion = nn.MSELoss()
    
    for ep in range(n_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
    # Evaluation
    model.eval()
    all_y_true, all_y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x.to(device))
            all_y_pred.append(y_normalizer.decode(out).cpu())
            all_y_true.append(y_normalizer.decode(y.to(device)).cpu())
            
    all_y_true = torch.cat(all_y_true, dim=0).numpy()
    all_y_pred = torch.cat(all_y_pred, dim=0).numpy()
    
    rel_l2 = np.linalg.norm(all_y_true - all_y_pred) / np.linalg.norm(all_y_true)
    print(f"Result for {wavelet_name}: RelL2 = {rel_l2:.6f}")
    return rel_l2

if __name__ == "__main__":
    results = {}
    wavelets = ['haar', 'db4', 'db6']
    for w in wavelets:
        results[w] = run_experiment(w, n_epochs=30)
        
    print("\n" + "="*30)
    print("FINAL WAVELET COMPARISON (30 Epochs)")
    for w, err in results.items():
        print(f"{w:8}: {err:.6f}")
    print("="*30)
