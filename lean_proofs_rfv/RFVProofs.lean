-- RFVProofs Lean proof project
-- Formal exploration of the "stable velocity field" diagnostic for
-- rectified-flow / velocity-prediction samplers. See README.md.
import RFVProofs.Defs           -- shared definitions (AffineTraj, finDiffVelocity, ...)
import RFVProofs.Stability      -- core diagnostic, fully proved (no sorry)
import RFVProofs.Straightness   -- literature-grounded anchor (explore targets)
import RFVProofs.Perturbation   -- quantitative drift bound (explore target)
import RFVProofs.LocalLinearity     -- windowed/multi-step generalization (explore target)
import RFVProofs.CompositeError     -- CLPC error budget: clip-weight decay, ODE bound, CFG drift
import RFVProofs.CorrectorConvergence -- Adams-Moulton corrector error bound
import RFVProofs.PECEOrderGain      -- PECE order gain: O(h^2) predictor → O(h^3) corrector
import RFVProofs.FlowModelCoord     -- λ(t)=t/(1-t) monotonicity + LocalLinearity transfer
import RFVProofs.EmbeddedPairError  -- corrector gap ≠ error signal; PECE consistency proof
import RFVProofs.RungeGuard         -- Lagrange blow-up bound; Euler predictor + AM corrector is unconditionally stable
import RFVProofs.AdamsStability     -- Full classification: explicit AB unstable (order≥2); implicit AM stable via partition of unity
import RFVProofs.WaveletDomain      -- Wavelet-domain error decomposition: Parseval + subband independence
import RFVProofs.HiddenMarkov       -- Markov-chain drift bounds: union bound + linear accumulation
import RFVProofs.KalmanFilter       -- Kalman prediction error bound + MMSE optimality (1D clean, matrix sorry)
import RFVProofs.ChebyshevAdaptive  -- Chebyshev minimax vs Adams Runge blow-up comparison
import RFVProofs.KalmanMatrixDesign -- Survival table for CLPC Kalman transition matrix A
import RFVProofs.GaussianProcessODE  -- GP implicit ODE: G-score Lyapunov certificate + entropy monotonicity
import RFVProofs.DoobSOC             -- Doob h-transform + SOC: score correction, PC as Feynman-Kac, Kalman blend
import RFVProofs.KLOptimality        -- KL divergence: log ≤ x-1, KL=0 iff equal, score step = KL descent
import RFVProofs.ProxySOCvsFull      -- Proxy SOC vs full SOC: same rate, constant overhead only
import RFVProofs.TokenSubspaceGuidance -- Token-level vanish/leak/bias guidance: subspace projection, 5-factor G-score, zero-extra-NFE cost model
import RFVProofs.MatrixTreeSoftOwnership -- Matrix-Tree (Koo et al. 2007) soft ownership: marginal partition-of-unity, soft leak generalization, cost model
import RFVProofs.VariableOrderGain     -- UniPC-style configurable max_order: general n-node AM partition of unity (Mathlib Lagrange), order-general truncation error gain, Chebyshev-gated safety for order > 3
import RFVProofs.TokenAvoidSOC         -- Token guidance as avoid-set SOC: exact good-sampler/good-token-set biconditional, terminal bad-set-avoidance from existing Lyapunov rate, intention-tree-as-constraint score additivity, non-Lipschitz argmax-ownership failure mode
