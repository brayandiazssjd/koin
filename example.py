import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import pyro.constraints as constraints

# ---- Fake data: 1 = heads, 0 = tails ----
data = torch.tensor([1., 0., 1., 1., 0., 1., 1., 1., 0., 1.])  # 7 heads, 3 tails

# ---- Model ----
def model(data):
    # Prior belief: θ ~ Uniform(0,1)
    theta = pyro.sample("theta", dist.Uniform(0., 1.))
    
    # Likelihood: flips are Bernoulli(θ)
    with pyro.plate("data", len(data)):
        pyro.sample("obs", dist.Bernoulli(theta), obs=data)

# ---- Guide (variational distribution for theta) ----
def guide(data):
    # Variational parameters for θ
    alpha_q = pyro.param("alpha_q", torch.tensor(1.), constraint=constraints.positive)
    beta_q  = pyro.param("beta_q", torch.tensor(1.), constraint=constraints.positive)
    
    # Approximate posterior: Beta distribution
    pyro.sample("theta", dist.Beta(alpha_q, beta_q))

# ---- Inference ----
pyro.clear_param_store()
optimizer = Adam({"lr": 0.02})
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

for step in range(1000):
    loss = svi.step(data)
    if step % 200 == 0:
        print(f"Step {step} Loss = {loss:.2f}")

# ---- Results ----
alpha_q = pyro.param("alpha_q").item()
beta_q = pyro.param("beta_q").item()
mean_theta = alpha_q / (alpha_q + beta_q)

print(f"\nPosterior alpha: {alpha_q:.2f}")
print(f"Posterior beta: {beta_q:.2f}")
print(f"Estimated P(heads) ≈ {mean_theta:.2f}")
