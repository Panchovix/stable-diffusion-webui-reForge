%% sure_gradient_symbolic.m
% Symbolic Math Toolbox — SURE gradient analysis for SGPS
% Paper: arXiv:2512.23232
%
% Three gradient modes in _sure_correct_x0 (sampling.py):
%   'approx' — stop-gradient:   grad = 2·r                            (no J)
%   'vjp'    — VJP of ||r||²:   grad = 2·(I − J_D)ᵀ·r               (no ∇tr J)
%   'full'   — full ∇SURE:      grad = 2·(I − J_D)ᵀ·r + 2σ²·∇tr{J_D}
%
% NOTE: Adam / AdamW is treated as a smart α controller only.
%       Its internal moments are not analysed here.

clc; clear; close all
fprintf('SURE Gradient Symbolic Analysis — arXiv:2512.23232\n')
fprintf(repmat('=',1,65)); fprintf('\n\n')

%% ═══════════════════════════════════════════════════════════════════
%  SECTION 1 — SCALAR CASE  (n = 1)
% ═══════════════════════════════════════════════════════════════════
fprintf('SECTION 1: SCALAR CASE  (n = 1)\n')
fprintf(repmat('-',1,65)); fprintf('\n')

syms x sigma real positive
syms D(x)

r1  = x - D(x);                    % residual
J1  = diff(D(x), x);               % Jacobian (scalar)

% ── SURE ─────────────────────────────────────────────────────────
SURE_1 = -sigma^2 + r1^2 + 2*sigma^2*J1;
fprintf('\nSURE (n=1):\n'); disp(SURE_1)

% ── Full ∇SURE via Symbolic Toolbox gradient() ───────────────────
grad_full_1 = gradient(SURE_1, x);
grad_full_1 = simplify(grad_full_1);
fprintf('∇_x SURE  (gradient(), full):\n'); disp(grad_full_1)

% ── Residual part:  gradient of ||r||² ───────────────────────────
grad_res_1  = gradient(r1^2, x);
grad_res_1  = simplify(grad_res_1);
fprintf('∇(||r||²) via gradient():\n'); disp(grad_res_1)

% ── Trace part:  gradient of 2σ²·tr{J_D} = 2σ²·J (n=1) ──────────
grad_tr_1   = gradient(2*sigma^2*J1, x);
grad_tr_1   = simplify(grad_tr_1);
fprintf('∇(2σ²·tr{J}) via gradient():\n'); disp(grad_tr_1)

% ── Three modes ───────────────────────────────────────────────────
fprintf('\n[approx]  grad = 2·r:\n');
disp(simplify(2*r1))

fprintf('[vjp]     grad = gradient(||r||², x)  =  2·(1−J)·r:\n');
disp(grad_res_1)

fprintf('[full]    grad = gradient(SURE, x):\n');
disp(grad_full_1)

fprintf('\nDifferences:\n')
fprintf('  [full]−[vjp]   = ∇(2σ²·tr{J}) =\n');
disp(simplify(grad_full_1 - grad_res_1))

fprintf('  [vjp]−[approx] = −2·J·r =\n');
disp(simplify(grad_res_1 - 2*r1))

%% ═══════════════════════════════════════════════════════════════════
%  SECTION 2 — VECTOR CASE  (n = 2)
% ═══════════════════════════════════════════════════════════════════
fprintf('\n'); fprintf(repmat('=',1,65)); fprintf('\n')
fprintf('SECTION 2: VECTOR CASE  (n = 2)\n')
fprintf(repmat('-',1,65)); fprintf('\n')

syms x1 x2 real
x_vec = [x1; x2];

syms D1(x1,x2) D2(x1,x2)
D_vec = [D1(x1,x2); D2(x1,x2)];

r_vec = x_vec - D_vec;             % residual vector
J_D   = jacobian(D_vec, x_vec);    % 2×2 Jacobian
fprintf('\nJ_D:\n'); disp(J_D)

% ── SURE ─────────────────────────────────────────────────────────
SURE_2 = -2*sigma^2 + r_vec.'*r_vec + 2*sigma^2*trace(J_D);
SURE_2 = simplify(SURE_2);
fprintf('\nSURE (n=2):\n'); disp(SURE_2)

% ── Full ∇SURE ────────────────────────────────────────────────────
grad_exact_2 = gradient(SURE_2, x_vec);
grad_exact_2 = simplify(grad_exact_2);
fprintf('∇_x SURE  (gradient(), full):\n'); disp(grad_exact_2)

% ── Residual part ────────────────────────────────────────────────
res_sq_2    = r_vec.' * r_vec;
grad_res_2  = gradient(res_sq_2, x_vec);
grad_res_2  = simplify(grad_res_2);
fprintf('∇(||r||²) = 2·(I−J_D)ᵀ·r  via gradient():\n'); disp(grad_res_2)

% ── Verify residual part equals 2·(I−J_D)ᵀ·r exactly ─────────────
check_res = simplify(grad_res_2 - 2*(eye(2) - J_D.')*r_vec);
fprintf('2·(I−J_D)ᵀ·r  −  gradient(||r||²):  (should be [0;0])\n')
disp(check_res)

% ── Trace part:  ∇(2σ²·tr{J_D}) via gradient() ───────────────────
% tr{J_D} is a function of x via J_D(x); gradient() differentiates through it
tr_J_D      = trace(J_D);                            % symbolic scalar
grad_tr_2   = gradient(2*sigma^2*tr_J_D, x_vec);
grad_tr_2   = simplify(grad_tr_2);
fprintf('∇(2σ²·tr{J_D})  via gradient():\n'); disp(grad_tr_2)

% ── Three modes ───────────────────────────────────────────────────
fprintf('[approx]  grad = 2·r:\n');
disp(simplify(2*r_vec))

fprintf('[vjp]     grad = gradient(||r||², x):\n');
disp(grad_res_2)

grad_full_2 = simplify(grad_res_2 + grad_tr_2);
fprintf('[full]    grad = gradient(||r||²) + gradient(2σ²·tr{J_D}):\n');
disp(grad_full_2)

% ── Verify full = exact ───────────────────────────────────────────
check_full = simplify(grad_exact_2 - grad_full_2);
fprintf('Verification  exact − full  (should be [0;0]):\n'); disp(check_full)

%% ═══════════════════════════════════════════════════════════════════
%  SECTION 3 — MC TRACE APPROXIMATION  (Hutchinson estimator)
% ═══════════════════════════════════════════════════════════════════
fprintf('\n'); fprintf(repmat('=',1,65)); fprintf('\n')
fprintf('SECTION 3: MC TRACE  b^T·J_D·b  →  tr{J_D}\n')
fprintf(repmat('-',1,65)); fprintf('\n')

% tr{J_D} ≈ (1/ε)·bᵀ·(D(x+εb)−D(x)),  b ~ N(0,I)
% First-order Taylor: D(x+εb) = D(x) + ε·J_D·b + O(ε²)
% ⟹  (1/ε)·bᵀ·(D(x+εb)−D(x)) ≈ bᵀ·J_D·b

syms b1 b2 real
b_vec  = [b1; b2];

% MC estimator (Taylor approximation)
tr_mc_taylor = simplify(b_vec.' * J_D * b_vec);
fprintf('\n(1/ε)·bᵀ·(D(x+εb)−D(x))  ≈  bᵀ·J_D·b  =\n'); disp(tr_mc_taylor)

% ── E_b[bᵀ·J·b] = tr(J) for b ~ N(0,I) ──────────────────────────
% Expand, then substitute E[b_i·b_j] = δ_{ij}:  b_i^2 → 1, b_i*b_j → 0 (i≠j)
tr_mc_exp = expand(tr_mc_taylor);
tr_mc_exp = subs(tr_mc_exp, b1^2, 1);
tr_mc_exp = subs(tr_mc_exp, b2^2, 1);
tr_mc_exp = subs(tr_mc_exp, b1*b2, 0);
tr_mc_exp = simplify(tr_mc_exp);
fprintf('E_b[bᵀ·J_D·b]  (expand then substitute E[b_i·b_j]=δ_{ij}):\n')
disp(tr_mc_exp)

fprintf('tr{J_D}  (direct):\n'); disp(simplify(tr_J_D))

check_mc = simplify(tr_mc_exp - tr_J_D);
fprintf('Difference  E[bᵀ·J_D·b] − tr{J_D}  (should be 0):\n'); disp(check_mc)

%% ═══════════════════════════════════════════════════════════════════
%  SECTION 4 — GRADIENT OF tr{J_D}  ('full' mode second term)
% ═══════════════════════════════════════════════════════════════════
fprintf('\n'); fprintf(repmat('=',1,65)); fprintf('\n')
fprintf('SECTION 4: ∇_x tr{J_D}  —  two approaches\n')
fprintf(repmat('-',1,65)); fprintf('\n')

% ── Approach A: gradient() directly on tr{J_D} ───────────────────
grad_tr_exact = gradient(tr_J_D, x_vec);
grad_tr_exact = simplify(grad_tr_exact);
fprintf('\nA) gradient(tr{J_D}, x)  (exact, via Symbolic Toolbox):\n')
disp(grad_tr_exact)

% ── Approach B: MC finite-difference of J_D^T·b  (as in 'full' mode code) ──
% In 'full' mode:
%   ∇tr{J_D}  ≈  (1/ε)·[J_D(x+εb)ᵀ·b − J_D(x)ᵀ·b]
% Taylor: J_D(x+εb) ≈ J_D(x) + ε·∑_k b_k·(∂J_D/∂x_k)

syms epsilon real positive

dJ_dx1 = diff(J_D, x1);            % ∂J_D/∂x1  (2×2 matrix)
dJ_dx2 = diff(J_D, x2);            % ∂J_D/∂x2  (2×2 matrix)

J_D_pert = J_D + epsilon*(b1*dJ_dx1 + b2*dJ_dx2);  % J_D(x+εb) to O(ε)

% MC gradient estimate: (1/ε)·[J_D(x+εb)ᵀ − J_D(x)ᵀ]·b
grad_tr_mc_raw = simplifyFraction((J_D_pert.' - J_D.') * b_vec / epsilon);
fprintf('B) (1/ε)·[J_D(x+εb)ᵀ−J_D(x)ᵀ]·b  (Taylor, ε cancels):\n')
disp(grad_tr_mc_raw)

% ── E_b[MC gradient] = ∇tr{J_D} ─────────────────────────────────
grad_tr_mc_exp = expand(grad_tr_mc_raw);
grad_tr_mc_exp = subs(grad_tr_mc_exp, b1^2, 1);
grad_tr_mc_exp = subs(grad_tr_mc_exp, b2^2, 1);
grad_tr_mc_exp = subs(grad_tr_mc_exp, b1*b2, 0);
grad_tr_mc_exp = simplify(grad_tr_mc_exp);
fprintf('E_b[MC gradient]  (substitute E[b_i·b_j]=δ_{ij}):\n')
disp(grad_tr_mc_exp)

check_tr = simplify(grad_tr_mc_exp - grad_tr_exact);
fprintf('Difference  E[MC grad] − gradient(tr{J_D})  (should be [0;0]):\n')
disp(check_tr)

%% ═══════════════════════════════════════════════════════════════════
%  SECTION 5 — LINEAR DENOISER  D(x) = W·x  (closed-form case)
% ═══════════════════════════════════════════════════════════════════
fprintf('\n'); fprintf(repmat('=',1,65)); fprintf('\n')
fprintf('SECTION 5: LINEAR DENOISER  D(x) = W·x\n')
fprintf(repmat('-',1,65)); fprintf('\n')

% For D(x) = W·x:  J_D = W  (constant),  ∇tr{J_D} = 0
% This isolates the pure residual gradient and shows 'full' = 'vjp'.

syms w11 w12 w21 w22 real
W = [w11 w12; w21 w22];

D_lin  = W * x_vec;
r_lin  = x_vec - D_lin;            % r = (I−W)·x
J_lin  = jacobian(D_lin, x_vec);   % = W  (constant)
fprintf('\nJ_D for D(x)=Wx:\n'); disp(J_lin)

% ── SURE ──────────────────────────────────────────────────────────
SURE_lin = -2*sigma^2 + r_lin.'*r_lin + 2*sigma^2*trace(J_lin);
SURE_lin = simplify(SURE_lin);
fprintf('SURE (linear D):\n'); disp(SURE_lin)

% ── ∇SURE via gradient() ──────────────────────────────────────────
grad_sure_lin = gradient(SURE_lin, x_vec);
grad_sure_lin = simplify(grad_sure_lin);
fprintf('∇SURE  (gradient(), linear D):\n'); disp(grad_sure_lin)

% ── Show it equals 2·(I−W)ᵀ·(I−W)·x ─────────────────────────────
IW = eye(2) - W;
grad_lin_factored = simplify(2 * IW.' * IW * x_vec);
fprintf('2·(I−W)ᵀ·(I−W)·x  (factored form):\n'); disp(grad_lin_factored)

check_lin = simplify(grad_sure_lin - grad_lin_factored);
fprintf('Difference  (should be [0;0]):\n'); disp(check_lin)

% ── Trace gradient is zero for linear D ───────────────────────────
tr_J_lin  = trace(J_lin);
grad_tr_lin = gradient(tr_J_lin, x_vec);
fprintf('∇tr{J_D} for linear D  (should be [0;0]  — J_D constant):\n')
disp(simplify(grad_tr_lin))

fprintf('⟹ For linear D:  [full] = [vjp]  (∇tr{J_D} = 0)\n\n')

% ── Three modes for linear D ──────────────────────────────────────
grad_approx_lin = simplify(2*r_lin);
grad_vjp_lin    = simplify(gradient(r_lin.'*r_lin, x_vec));
grad_full_lin   = simplify(grad_vjp_lin + gradient(2*sigma^2*tr_J_lin, x_vec));

fprintf('[approx]  2·r = 2·(I−W)·x:\n');   disp(grad_approx_lin)
fprintf('[vjp]     gradient(||r||²):\n');    disp(grad_vjp_lin)
fprintf('[full]    gradient(SURE):\n');       disp(grad_full_lin)

diff_vjp_approx_lin = simplify(grad_vjp_lin - grad_approx_lin);
fprintf('[vjp]−[approx] = −2·Wᵀ·(I−W)·x:\n'); disp(diff_vjp_approx_lin)

%% ═══════════════════════════════════════════════════════════════════
%  SECTION 6 — EFFECTIVE STEP SIZE  (α controller perspective)
% ═══════════════════════════════════════════════════════════════════
fprintf('\n'); fprintf(repmat('=',1,65)); fprintf('\n')
fprintf('SECTION 6: EFFECTIVE STEP SIZE  α_eff = α / (1 + σ_t)\n')
fprintf(repmat('-',1,65)); fprintf('\n')

syms alpha sigma_t real positive

alpha_eff = alpha / (1 + sigma_t);
fprintf('\nα_eff = α/(1+σ_t) =\n'); disp(alpha_eff)

% Gradient update:  x* = x − α_eff · ∇SURE
% Substituting α_eff into the full-mode gradient update (scalar n=1):
syms g_sure real   % placeholder for any of the three gradient modes
x_star = x - alpha_eff * g_sure;
fprintf('x* = x − α_eff·∇SURE =\n'); disp(x_star)

% ── Effective update size relative to raw gradient ────────────────
fprintf('α_eff / α  =  1/(1+σ_t):\n')
disp(simplify(alpha_eff / alpha))

fprintf('Limits:\n')
fprintf('  σ_t → 0  (clean image):   α_eff → '); disp(limit(alpha_eff, sigma_t, 0))
fprintf('  σ_t → ∞  (pure noise):    α_eff → '); disp(limit(alpha_eff, sigma_t, inf))

% ── KL contraction factor (Theorem 2 / β_t term) ─────────────────
% β_t = α · σ̂₀²   (effective LR in the KL convergence proof)
syms sigma_hat_0 mu real positive
beta_t  = alpha * sigma_hat_0^2;
contraction = 1 - beta_t * mu;
fprintf('\nKL contraction factor  (1 − β_t·μ),  β_t = α·σ̂₀²:\n')
disp(simplify(contraction))

% Fixed-point condition: contraction < 1
fp_cond = solve(contraction < 1, alpha);
fprintf('Contraction < 1  requires  α > :\n'); disp(fp_cond)

% Upper bound β_t ≤ 1/(2L) from Assumption 3
syms L real positive
beta_upper  = 1 / (2*L);
alpha_upper = simplify(beta_upper / sigma_hat_0^2);
fprintf('From β_t ≤ 1/(2L):  α ≤ 1/(2L·σ̂₀²) =\n'); disp(alpha_upper)

fprintf('\n'); fprintf(repmat('=',1,65)); fprintf('\n')
fprintf('Done.  Return output; I will read the CAS results.\n')
fprintf(repmat('=',1,65)); fprintf('\n')

%% ═══════════════════════════════════════════════════════════════════
%  SECTION 7 — EXPORT TO LaTeX
% ═══════════════════════════════════════════════════════════════════
% Collect every named result computed above and write a self-contained
% .tex file.  Each entry in `results` is a struct with fields:
%   .label   — short identifier used as equation label
%   .caption — human-readable description printed above the equation
%   .expr    — symbolic expression (scalar or column vector)

results = {};

% ── Section 1: scalar ────────────────────────────────────────────
results{end+1} = struct('label','sure_scalar', ...
    'caption','SURE formula  (n=1)', ...
    'expr', SURE_1);

results{end+1} = struct('label','grad_sure_scalar_full', ...
    'caption','Full $\nabla_x \mathrm{SURE}$  (n=1, gradient())', ...
    'expr', grad_full_1);

results{end+1} = struct('label','grad_res_scalar', ...
    'caption','Residual part $\nabla\|r\|^2$  (n=1)', ...
    'expr', grad_res_1);

results{end+1} = struct('label','grad_tr_scalar', ...
    'caption','Trace part $\nabla(2\sigma^2 \operatorname{tr}\{J_D\})$  (n=1)', ...
    'expr', grad_tr_1);

results{end+1} = struct('label','grad_approx_scalar', ...
    'caption','[approx] mode gradient  (n=1): $2r$', ...
    'expr', simplify(2*r1));

results{end+1} = struct('label','grad_vjp_scalar', ...
    'caption','[vjp] mode gradient  (n=1): $2(1-J)r$', ...
    'expr', grad_res_1);

results{end+1} = struct('label','diff_full_vjp_scalar', ...
    'caption','[full]$-$[vjp]  (n=1):  $2\sigma^2 D''''(x)$', ...
    'expr', simplify(grad_full_1 - grad_res_1));

results{end+1} = struct('label','diff_vjp_approx_scalar', ...
    'caption','[vjp]$-$[approx]  (n=1):  $-2J \cdot r$', ...
    'expr', simplify(grad_res_1 - 2*r1));

% ── Section 2: vector ────────────────────────────────────────────
results{end+1} = struct('label','sure_vec', ...
    'caption','SURE formula  (n=2)', ...
    'expr', SURE_2);

results{end+1} = struct('label','grad_sure_vec_full', ...
    'caption','Full $\nabla_x \mathrm{SURE}$  (n=2, gradient())', ...
    'expr', grad_exact_2);

results{end+1} = struct('label','grad_res_vec', ...
    'caption','$\nabla\|r\|^2 = 2(I-J_D)^\top r$  (n=2)', ...
    'expr', grad_res_2);

results{end+1} = struct('label','grad_tr_vec', ...
    'caption','$\nabla(2\sigma^2\operatorname{tr}\{J_D\})$  (n=2)', ...
    'expr', grad_tr_2);

results{end+1} = struct('label','check_res_vec', ...
    'caption','Verification: $\nabla\|r\|^2 - 2(I-J_D)^\top r$  (should be $\mathbf{0}$)', ...
    'expr', check_res);

results{end+1} = struct('label','grad_full_vec', ...
    'caption','[full] mode gradient  (n=2)', ...
    'expr', grad_full_2);

results{end+1} = struct('label','check_full_vec', ...
    'caption','Verification: exact$-$full  (should be $\mathbf{0}$)', ...
    'expr', check_full);

% ── Section 3: MC trace ──────────────────────────────────────────
results{end+1} = struct('label','mc_trace_taylor', ...
    'caption','MC trace estimate $b^\top J_D b$ (Taylor)', ...
    'expr', tr_mc_taylor);

results{end+1} = struct('label','mc_trace_expected', ...
    'caption','$\mathbb{E}_b[b^\top J_D b]$  (substitute $\mathbb{E}[b_i b_j]=\delta_{ij}$)', ...
    'expr', tr_mc_exp);

results{end+1} = struct('label','check_mc_trace', ...
    'caption','Verification: $\mathbb{E}[b^\top J_D b] - \operatorname{tr}\{J_D\}$  (should be $0$)', ...
    'expr', check_mc);

% ── Section 4: gradient of trace ────────────────────────────────
results{end+1} = struct('label','grad_tr_exact', ...
    'caption','$\nabla_x \operatorname{tr}\{J_D\}$  via gradient()  (exact)', ...
    'expr', grad_tr_exact);

results{end+1} = struct('label','grad_tr_mc_raw', ...
    'caption','MC finite-difference  $(1/\varepsilon)[J_D(x{+}\varepsilon b)^\top{-}J_D(x)^\top]b$  (Taylor, $\varepsilon$ cancelled)', ...
    'expr', grad_tr_mc_raw);

results{end+1} = struct('label','grad_tr_mc_expected', ...
    'caption','$\mathbb{E}_b[\text{MC grad trace}]$  (substitute $\mathbb{E}[b_i b_j]=\delta_{ij}$)', ...
    'expr', grad_tr_mc_exp);

results{end+1} = struct('label','check_tr_grad', ...
    'caption','Verification: $\mathbb{E}[\text{MC grad}] - \nabla\operatorname{tr}\{J_D\}$  (should be $\mathbf{0}$)', ...
    'expr', check_tr);

% ── Section 5: linear denoiser ───────────────────────────────────
results{end+1} = struct('label','sure_linear', ...
    'caption','SURE for linear denoiser $D(x)=Wx$', ...
    'expr', SURE_lin);

results{end+1} = struct('label','grad_sure_linear', ...
    'caption','$\nabla\mathrm{SURE}$ for linear $D$  via gradient()', ...
    'expr', grad_sure_lin);

results{end+1} = struct('label','grad_sure_linear_factored', ...
    'caption','$\nabla\mathrm{SURE}$ factored: $2(I-W)^\top(I-W)x$', ...
    'expr', grad_lin_factored);

results{end+1} = struct('label','grad_tr_linear', ...
    'caption','$\nabla\operatorname{tr}\{J_D\}$ for linear $D$  (should be $\mathbf{0}$)', ...
    'expr', simplify(grad_tr_lin));

results{end+1} = struct('label','diff_vjp_approx_linear', ...
    'caption','[vjp]$-$[approx] for linear $D$: $-2W^\top(I-W)x$', ...
    'expr', diff_vjp_approx_lin);

% ── Section 6: effective step size ───────────────────────────────
results{end+1} = struct('label','alpha_eff', ...
    'caption','Effective step size $\alpha_\mathrm{eff} = \alpha/(1+\sigma_t)$', ...
    'expr', alpha_eff);

results{end+1} = struct('label','contraction', ...
    'caption','KL contraction factor $(1 - \beta_t \mu)$,  $\beta_t = \alpha\hat{\sigma}_0^2$', ...
    'expr', contraction);

results{end+1} = struct('label','alpha_upper', ...
    'caption','Upper bound on $\alpha$ from $\beta_t \le 1/(2L)$', ...
    'expr', alpha_upper);

% ── Write the .tex file ───────────────────────────────────────────
tex_path = '/Users/nickzeng/Documents/GitHub/stable-diffusion-webui-reForge/sure_gradient_results.tex';
fid = fopen(tex_path, 'w');
if fid == -1
    error('Cannot open %s for writing.', tex_path);
end

% Preamble
fprintf(fid, '\\documentclass{article}\n');
fprintf(fid, '\\usepackage{amsmath,amssymb,geometry,booktabs,xcolor}\n');
fprintf(fid, '\\geometry{margin=2cm}\n');
fprintf(fid, '\\title{SURE Gradient --- Symbolic CAS Results\\\\');
fprintf(fid, '{\\large arXiv:2512.23232 \\quad \\texttt{sure\\_gradient\\_symbolic.m}}}\n');
fprintf(fid, '\\date{\\today}\n');
fprintf(fid, '\\begin{document}\n\\maketitle\n\n');
fprintf(fid, '\\noindent All expressions below are exact outputs of MATLAB Symbolic\n');
fprintf(fid, 'Math Toolbox.  No manual simplification has been applied.\n\n');

% One subsection per original script section
section_titles = { ...
    'Scalar case ($n=1$)', ...
    'Vector case ($n=2$)', ...
    'MC trace approximation (Hutchinson estimator)', ...
    "Gradient of $\operatorname{tr}\{J_D\}$ --- ``full'' mode second term", ...
    'Linear denoiser $D(x)=Wx$ (closed-form special case)', ...
    'Effective step size ($\alpha$ controller perspective)'};

% section_breaks(s) = index of first result belonging to section s.
% Count:  S1=8, S2=7, S3=3, S4=4, S5=5, S6=3  →  cumsum: 1,9,16,19,23,28,31
section_breaks = [1, 9, 16, 19, 23, 28, 31];

for s = 1:numel(section_titles)
    fprintf(fid, '\\section{%s}\n\n', section_titles{s});
    idx_start = section_breaks(s);
    idx_end   = section_breaks(s+1) - 1;

    for k = idx_start:idx_end
        R   = results{k};
        tex_expr = latex(R.expr);

        fprintf(fid, '\\noindent\\textbf{%s}\n', R.caption);
        fprintf(fid, '\\begin{align}\n');

        % Vector results: split into component equations
        if ~isscalar(R.expr) && isvector(R.expr)
            n_comp = numel(R.expr);
            for c = 1:n_comp
                comp_tex = latex(R.expr(c));
                if c < n_comp
                    fprintf(fid, '  %s \\\\\n', comp_tex);
                else
                    fprintf(fid, '  %s\n', comp_tex);
                end
            end
            fprintf(fid, '  \\tag{\\texttt{%s}}\n', strrep(R.label,'_','-'));
        else
            fprintf(fid, '  %s \\tag{\\texttt{%s}}\n', ...
                tex_expr, strrep(R.label,'_','-'));
        end

        fprintf(fid, '\\end{align}\n\n');
    end
end

% Closing
fprintf(fid, '\\end{document}\n');
fclose(fid);

fprintf('\nLaTeX output written to:\n  %s\n', tex_path)
fprintf('Compile with:  pdflatex sure_gradient_results.tex\n')
