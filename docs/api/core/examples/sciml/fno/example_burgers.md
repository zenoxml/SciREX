# Fourier Neural Operator (FNO) - Training Tutorial - Burgers Equation

## Mathematical Formulation

The 1D Burgers equation:

$$
\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} = \nu\frac{\partial^2 u}{\partial x^2}
$$

where:
- $u(x,t)$: velocity field
- $\nu$: viscosity coefficient
- $x$: spatial coordinate
- $t$: time

## Implementation Parameters

### Domain Parameters
```python
x_domain = [0, 2π]
spatial_resolution = 8192  # Full resolution
training_resolution = 256  # Subsampled for training
```

### Dataset Parameters
```python
train_samples = 1000
test_samples = 200
batch_size = 100
epochs = 200
```

### FNO Architecture
```python
fno = FNO1d(
    in_channels=2,    # Initial condition + spatial coordinate
    out_channels=1,   # Solution at t=1
    modes=16,         # Number of Fourier modes
    width=64,         # Channel width
    activation=jax.nn.relu,
    n_blocks=4
)
```

## Implementation Steps

### 1. Data Loading
```python
data = scipy.io.loadmat("burgers_data_R10.mat")
a, u = data["a"], data["u"]  # Initial conditions and solutions

# Add channel dimension and mesh
a = a[:, jnp.newaxis, :]
u = u[:, jnp.newaxis, :]
mesh = jnp.linspace(0, 2 * jnp.pi, u.shape[-1])
```

### 2. Data Preprocessing
```python
# Combine initial condition with mesh information
mesh_shape_corrected = jnp.repeat(mesh[jnp.newaxis, jnp.newaxis, :], u.shape[0], axis=0)
a_with_mesh = jnp.concatenate((a, mesh_shape_corrected), axis=1)

# Train-test split
train_x, test_x = a_with_mesh[:1000], a_with_mesh[1000:1200]
train_y, test_y = u[:1000], u[1000:1200]
```

### 3. Training Loop
```python
@eqx.filter_jit
def make_step(model, state, x, y):
    loss, grad = eqx.filter_value_and_grad(loss_fn)(model, x, y)
    val_loss = loss_fn(model, test_x[..., ::32], test_y[..., ::32])
    updates, new_state = optimizer.update(grad, state, model)
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_state, loss, val_loss
```

### 4. Evaluation Metrics
```python
def relative_l2_norm(pred, ref):
    diff_norm = jnp.linalg.norm(pred - ref)
    ref_norm = jnp.linalg.norm(ref)
    return diff_norm / ref_norm
```

## Results Analysis

The implementation generates five visualizations:

1. `initial_vs_after.png`: Initial condition vs solution at t=1
2. `loss.png`: Training and validation loss curves
3. `prediction.png`: Model prediction vs ground truth
4. `difference.png`: Error analysis
5. `superresolution.png`: Zero-shot superresolution capability

### Key Performance Metrics
- Relative L2 error: ~1e-2
- Training time: 200 epochs
- Resolution invariance demonstrated through superresolution

## Output Directory Structure
```
outputs/fno/burgers/
├── initial_vs_after.png
├── loss.png
├── prediction.png
├── difference.png
└── superresolution.png
```