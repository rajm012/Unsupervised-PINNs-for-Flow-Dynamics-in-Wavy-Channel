
# Dataset Structure
```text
points/
├── sym/
│   ├── boundary.csv
│   └── collocation.csv
├── phi_30/
│   ├── boundary.csv
│   └── collocation.csv
├── phi_60/
├── phi_90/
├── phi_180/
├── phi_-30/
├── phi_-60/
└── phi_-90/
```

Each folder corresponds to one channel geometry defined by its phase shift: `sym`[Symmetric channel (ϕ = 0°)], `phi_30`, `phi_60`, `phi_90`, `phi_180`, `phi_-30`, `phi_-60`, `phi_-90`

# Files Generated for Each Geometry
Each geometry contains **two CSV files** and **one visualization**.

## 1. `collocation.csv`
Contains the interior collocation points used to enforce the governing PDE residuals during PINN training. These points are generated using Latin Hypercube Sampling with additional refinement near the channel boundaries.

### Number of points
| Geometry                            |     Points |
| ----------------------------------- | ---------: |
| Symmetric                           | **62,000** |
| Asymmetric (±30°, ±60°, ±90°, 180°) | **89,600** |

### Columns
| Column | Description                           |
| ------ | ------------------------------------- |
| `x`    | x-coordinate of the collocation point |
| `y`    | y-coordinate of the collocation point |
| `zone` | Sampling region (`core` or `bl`)      |

where
* **core** → Interior flow region
* **bl** → Boundary-layer refined region near walls/inlet/outlet


## 2. `boundary.csv`
Contains all boundary points used to impose the PINN boundary conditions.

### Number of points
| Geometry                            |    Points |
| ----------------------------------- | --------: |
| Symmetric                           | **8,000** |
| Asymmetric (±30°, ±60°, ±90°, 180°) | **9,000** |

### Columns
| Column | Description                            |
| ------ | -------------------------------------- |
| `x`    | x-coordinate                           |
| `y`    | y-coordinate                           |
| `nx`   | x-component of the outward unit normal |
| `ny`   | y-component of the outward unit normal |
| `type` | Boundary classification                |

Possible values of `type` are: `upper_wall`, `lower_wall`, `inlet`, `outlet`.
The outward unit normals (`nx`, `ny`) are computed analytically and can be directly used for Neumann boundary conditions, heat-flux calculations, or wall shear stress evaluation.

# Geometry Coverage
The dataset includes the following eight channel configurations:
| Geometry   | Phase Shift |
| ---------- | ----------: |
| Symmetric  |          0° |
| Asymmetric |        +30° |
| Asymmetric |        +60° |
| Asymmetric |        +90° |
| Asymmetric |       +180° |
| Asymmetric |        −30° |
| Asymmetric |        −60° |
| Asymmetric |        −90° |

# Purpose of Each Dataset
* **`collocation.csv`** provides the interior points where the PINN evaluates the governing equations (e.g., continuity, momentum, and later the energy equation).
* **`boundary.csv`** provides the points where the prescribed boundary conditions are enforced (inlet, outlet, and wall conditions).
