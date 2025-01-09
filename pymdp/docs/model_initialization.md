# RGM Model Initialization

## Matrix Loading

The RGM model requires three types of matrices for each level:

1. Recognition Matrices (R)
   - Bottom-up information flow
   - Loaded from R0.npy, R1.npy, R2.npy
   - Shape constraints match hierarchy dimensions

2. Generative Matrices (G)
   - Top-down information flow
   - Loaded from G0.npy, G1.npy, G2.npy
   - Transposed shapes of recognition matrices

3. Lateral Matrices (L)
   - Within-level connections
   - Loaded from L0.npy, L1.npy, L2.npy
   - Square matrices for each level

## Validation Rules

The model initializer enforces several constraints:

1. Matrix Presence
   - All required matrices must exist
   - Naming follows R/G/L prefix convention
   - Three matrices of each type (levels 0-2)

2. Shape Relationships
   - Recognition/Generative pairs are transposed
   - Lateral matrices are square
   - Dimensions match hierarchy specification

3. Numeric Properties
   - All matrices are loaded as torch.Tensor
   - Moved to specified device (CPU/CUDA)
   - Preserved as float32 precision 