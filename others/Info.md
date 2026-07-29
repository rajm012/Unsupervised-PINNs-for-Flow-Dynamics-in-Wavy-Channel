----------------------------------------------------------

- Collect the `collocation points` via `Latin Hypercube Sampling`.
- No need to collect the velocity and pressure data for now
- We need to collect the Cauchy stress tensors(sigma) along with the velocity(shi-> u,v) and pressure(p)
- Not standard 2nd order NS-Eqn, using the mixed variable approach
- Incorporate the 2d energy equation to replicate thermal analysis

----------------------------------------------------------

Networks:
- Input (x,y)
- Output (shi:-stream fxn, p, sigma:-Cauchy stress, temp)
- 5*80 vs 6*100, tanh
- Initial Learning Rate: 0.001
- Rate Scheduler: gamma=0.8, n(t+1)=max(n(min), n(t)*gamma), n(min)=0.000001
- change LR when loss doesn't change for the 20 cts epochs
- Adam Optimizer with Xavier as initial
- Re = 10, 80 for all
- phi = 30, 60, 90, 120

plane vs sym vs asym: 
- arch: 5*80 vs 6*100 vs 6*100
- col-points: 39.6k vs 62k vs 89.6k
- bc-points: 8k vs 8k vs 9k

----------------------------------------------------------

Trained on: 
- NS Residual
- Continuity Residual
- Energy equation Residual
- Boundary Condition losses
- No data loss

----------------------------------------------------------
after replication we have to: (x, y, Re, Phase Angle) vs (x, y)

----------------------------------------------------------
