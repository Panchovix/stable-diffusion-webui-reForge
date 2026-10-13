-- imports MUST precede all content in Lean 4
import SureGPU.Types

/-!
# Buffer Lifespan Theory

A GPU buffer occupies device memory for the interval `[created, freed)`.
Using the *tight* lifespan — freeing the buffer one step after its last use —
minimises peak memory and maximises the opportunity to reuse allocations.
-/

namespace SureGPU

/-!
## §1  Basic Span Lemmas
-/

theorem Span.liveAt_iff (s : Span) (t : Step) :
    s.liveAt t ↔ s.start ≤ t ∧ t < s.finish := Iff.rfl

theorem Span.not_liveAt_before {s : Span} {t : Step} (h : t < s.start) : ¬ s.liveAt t :=
  fun ⟨h1, _⟩ => Nat.lt_irrefl t (Nat.lt_of_lt_of_le h h1)

theorem Span.not_liveAt_after {s : Span} {t : Step} (h : s.finish ≤ t) : ¬ s.liveAt t :=
  fun ⟨_, h2⟩ => Nat.lt_irrefl t (Nat.lt_of_lt_of_le h2 h)

/-- A span is live at its start when `start < finish` (strictly non-empty). -/
theorem Span.liveAt_start {s : Span} (hv : s.start < s.finish) : s.liveAt s.start :=
  ⟨le_refl _, hv⟩

/-- The tight span is live at the last-used step. -/
theorem Span.tight_liveAt_lastUsed (c u : Step) (h : c ≤ u) :
    (Span.tight c u).liveAt u :=
  ⟨h, Nat.lt_succ_self u⟩

/-!
## §2  The Tight Span is the Minimum-Length Valid Span
-/

/-- The tight freed time equals `lastUsed + 1`. -/
theorem Span.tight_finish_eq (c u : Step) : (Span.tight c u).finish = u + 1 := rfl

/-- Any freed time must be at least `lastUsed + 1`. -/
theorem Span.freed_at_least {c u f : Step} (h : u < f) :
    (Span.tight c u).finish ≤ f := h

/-- The tight span is strictly shorter than any span with a later freed time. -/
theorem Span.tight_is_minimal {c u f : Step} (h_cu : c ≤ u) (h_late : u + 1 < f) :
    (Span.tight c u).length < ({ start := c, finish := f } : Span).length :=
  -- Span.length and Span.tight are @[reducible]; the goal reduces to u+1-c < f-c.
  -- Nat.sub_lt_sub_right: k ≤ n → n < m → n - k < m - k
  Nat.sub_lt_sub_right (Nat.le_succ_of_le h_cu) h_late

/-- Memory wasted = number of extra live steps beyond the tight span. -/
def Span.waste (s : Span) (lastUsed : Step) : ℕ :=
  s.finish - (lastUsed + 1)

theorem Span.tight_zero_waste (c u : Step) : (Span.tight c u).waste u = 0 :=
  Nat.sub_self _

theorem Span.waste_eq (s : Span) (u : Step) : s.waste u = s.finish - (u + 1) := rfl

/-!
## §3  Disjointness
-/

def Span.disjoint (a b : Span) : Prop := ¬ a.overlaps b

theorem Span.disjoint_not_both_live {a b : Span} (h : a.disjoint b) (t : Step) :
    ¬ (a.liveAt t ∧ b.liveAt t) :=
  fun ⟨ha, hb⟩ => h ⟨t, ha, hb⟩

theorem Span.disjoint_can_share {a b : Span} (h : a.disjoint b) :
    ∀ t : Step, ¬ (a.liveAt t ∧ b.liveAt t) :=
  Span.disjoint_not_both_live h

theorem Span.disjoint_comm (a b : Span) : a.disjoint b ↔ b.disjoint a := by
  simp only [Span.disjoint, Span.overlaps]
  constructor <;> intro h ⟨t, ha, hb⟩ <;> exact h ⟨t, hb, ha⟩

/-- If `a.finish ≤ b.start`, the spans cannot overlap. -/
theorem Span.disjoint_of_finish_le_start {a b : Span} (h : a.finish ≤ b.start) :
    a.disjoint b :=
  fun ⟨t, ⟨_, h2⟩, ⟨h3, _⟩⟩ =>
    Nat.lt_irrefl t (Nat.lt_of_lt_of_le (Nat.lt_of_lt_of_le h2 h) h3)

/-!
## §4  SURE Computation Step Constants

Using `abbrev` makes these transparent to Lean 4's `decide` / instance synthesis.
-/

abbrev STEP_PARALLEL_1 : Step := 1  -- genB [S2] ∥ denoiser1 [S1]
abbrev STEP_SYNC       : Step := 2  -- cudaStreamSynchronize barrier
abbrev STEP_PERTURB    : Step := 3  -- x_perturbed = x_noisy + ε·b
abbrev STEP_DENOISER2  : Step := 4  -- D_θ(x_perturbed, σ)
abbrev STEP_JAC_TRACE  : Step := 5  -- b·(r₂ − r₁)/ε
abbrev STEP_SURE       : Step := 6  -- SURE formula
abbrev STEP_GRAD_D2H   : Step := 7  -- computeGrad ∥ async D2H

abbrev STEP_GEN_B     := STEP_PARALLEL_1
abbrev STEP_DENOISER1 := STEP_PARALLEL_1

/-!
## §5  Concrete Spans for Each SURE Tensor
-/

abbrev span_b        : Span := Span.tight STEP_GEN_B     STEP_JAC_TRACE  -- [1, 6)
abbrev span_r1       : Span := Span.tight STEP_DENOISER1 STEP_SURE        -- [1, 7)
abbrev span_xp       : Span := Span.tight STEP_PERTURB   STEP_DENOISER2   -- [3, 5)
abbrev span_r2       : Span := Span.tight STEP_DENOISER2 STEP_JAC_TRACE   -- [4, 6)
abbrev span_jt       : Span := Span.tight STEP_JAC_TRACE STEP_SURE        -- [5, 7)
abbrev span_sureVal  : Span := Span.tight STEP_SURE      STEP_GRAD_D2H    -- [6, 8)
abbrev span_sureGrad : Span := Span.tight STEP_GRAD_D2H  (STEP_GRAD_D2H + 1)  -- [7, 8)

-- Validity (all evaluated by decide since everything is an abbrev)
theorem span_b_valid        : span_b.valid        := by decide
theorem span_r1_valid       : span_r1.valid       := by decide
theorem span_xp_valid       : span_xp.valid       := by decide
theorem span_r2_valid       : span_r2.valid       := by decide
theorem span_jt_valid       : span_jt.valid       := by decide
theorem span_sureVal_valid  : span_sureVal.valid  := by decide
theorem span_sureGrad_valid : span_sureGrad.valid := by decide

/-!
## §6  Memory-Sharing Opportunities
-/

theorem span_xp_disjoint_jt : span_xp.disjoint span_jt :=
  Span.disjoint_of_finish_le_start (by decide)

theorem span_r2_disjoint_sureVal : span_r2.disjoint span_sureVal :=
  Span.disjoint_of_finish_le_start (by decide)

theorem span_xp_disjoint_sureVal : span_xp.disjoint span_sureVal :=
  Span.disjoint_of_finish_le_start (by decide)

theorem span_xp_disjoint_sureGrad : span_xp.disjoint span_sureGrad :=
  Span.disjoint_of_finish_le_start (by decide)

theorem span_jt_disjoint_sureGrad : span_jt.disjoint span_sureGrad :=
  Span.disjoint_of_finish_le_start (by decide)

theorem sure_memory_sharing_pairs :
    span_xp.disjoint span_jt ∧
    span_r2.disjoint span_sureVal ∧
    span_jt.disjoint span_sureGrad :=
  ⟨span_xp_disjoint_jt, span_r2_disjoint_sureVal, span_jt_disjoint_sureGrad⟩

/-!
## §7  All Tensor Spans are Tight (Zero Waste)
-/

theorem span_b_tight        : span_b.waste STEP_JAC_TRACE       = 0 := by decide
theorem span_r1_tight       : span_r1.waste STEP_SURE            = 0 := by decide
theorem span_xp_tight       : span_xp.waste STEP_DENOISER2       = 0 := by decide
theorem span_r2_tight       : span_r2.waste STEP_JAC_TRACE        = 0 := by decide
theorem span_jt_tight       : span_jt.waste STEP_SURE             = 0 := by decide
theorem span_sureVal_tight  : span_sureVal.waste STEP_GRAD_D2H    = 0 := by decide

theorem all_sure_spans_tight :
    span_b.waste STEP_JAC_TRACE       = 0 ∧
    span_r1.waste STEP_SURE            = 0 ∧
    span_xp.waste STEP_DENOISER2       = 0 ∧
    span_r2.waste STEP_JAC_TRACE        = 0 ∧
    span_jt.waste STEP_SURE             = 0 ∧
    span_sureVal.waste STEP_GRAD_D2H    = 0 :=
  ⟨span_b_tight, span_r1_tight, span_xp_tight,
   span_r2_tight, span_jt_tight, span_sureVal_tight⟩

end SureGPU
