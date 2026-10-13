-- imports MUST precede all content (including doc comments) in Lean 4
import Mathlib.Data.Nat.Basic
import Mathlib.Data.List.Basic

/-!
# Core Types for GPU SURE Computation

This module defines the abstract type vocabulary for reasoning about:
1. **Buffer lifespans** — when device memory is allocated, live, and freed.
2. **CUDA stream parallelism** — which operations execute concurrently.
3. **Transfer policy** — what must cross the PCIe bus between Host and Device.

The model is *abstract*: it does not bind to actual CUDA C++ but gives a
mathematical skeleton that the scheduling proofs in later modules rely on.

## Design conventions
- Time is discrete (`Step = ℕ`), one increment per kernel launch or transfer.
- Intervals are half-open `[start, finish)`.
- Each GPU kernel reads a list of existing buffers and writes exactly one new one
  (single-output; multi-output ops are split).
-/

namespace SureGPU

/-!
## §1  Discrete Time
-/

/-- Discrete clock tick.  Each kernel launch or transfer consumes one step. -/
abbrev Step := ℕ

/-!
## §2  Memory Locations
-/

/-- The two sides of the PCIe bus. -/
inductive MemLoc : Type where
  | Host   : MemLoc   -- CPU system RAM
  | Device : MemLoc   -- GPU VRAM
  deriving DecidableEq, Repr, BEq

/-- Switching sides gives the other location. -/
def MemLoc.flip : MemLoc → MemLoc
  | MemLoc.Host   => MemLoc.Device
  | MemLoc.Device => MemLoc.Host

@[simp] theorem MemLoc.flip_host   : MemLoc.Host.flip   = MemLoc.Device := rfl
@[simp] theorem MemLoc.flip_device : MemLoc.Device.flip = MemLoc.Host   := rfl
@[simp] theorem MemLoc.flip_flip   : ∀ l : MemLoc, l.flip.flip = l := by
  intro l; cases l <;> rfl

/-- A transfer always moves between *distinct* locations. -/
theorem MemLoc.flip_ne (l : MemLoc) : l.flip ≠ l := by cases l <;> decide

/-!
## §3  CUDA Streams
-/

/-- A CUDA stream is identified by a non-negative integer.
    Stream 0 is the default stream (synchronises with all others). -/
structure StreamId where
  id : ℕ
  deriving DecidableEq, Repr, BEq

/-- Conventional stream names used throughout the SURE schedule. -/
def S0 : StreamId := ⟨0⟩  -- default (null) stream
def S1 : StreamId := ⟨1⟩  -- main compute stream
def S2 : StreamId := ⟨2⟩  -- cuRAND random-generation stream
def S3 : StreamId := ⟨3⟩  -- asynchronous PCIe copy stream

theorem S1_ne_S2 : S1 ≠ S2 := by decide
theorem S1_ne_S3 : S1 ≠ S3 := by decide
theorem S2_ne_S3 : S2 ≠ S3 := by decide

/-!
## §4  Buffer Descriptor

A *buffer* is a chunk of memory on one side of the PCIe bus.
Its size is measured in 32-bit floats (4 bytes each).
-/

/-- Abstract descriptor for a GPU/CPU memory region. -/
structure Buffer where
  numel : ℕ       -- number of float32 elements
  loc   : MemLoc  -- which side it currently lives on
  deriving Repr

/-- Size in bytes (float32 = 4 bytes). -/
def Buffer.sizeBytes (b : Buffer) : ℕ := 4 * b.numel

/-- A buffer on the device doesn't require a Host→Device transfer. -/
def Buffer.onDevice (b : Buffer) : Prop := b.loc = MemLoc.Device

/-- A buffer on the host doesn't require a Device→Host transfer. -/
def Buffer.onHost (b : Buffer) : Prop := b.loc = MemLoc.Host

/-!
## §5  Spans (Half-Open Intervals)

A *span* `[start, finish)` describes how long a buffer stays alive.
The buffer is *live* at step `t` iff `start ≤ t < finish`.
The *tight* span frees the buffer on the step *immediately after* its last use,
minimising peak device memory consumption.
-/

/-- Half-open interval [start, finish). -/
structure Span where
  start  : Step
  finish : Step
  deriving Repr, BEq

/-- A span is *valid* when it is non-empty (start ≤ finish). -/
@[reducible] def Span.valid (s : Span) : Prop := s.start ≤ s.finish

/-- A buffer is live at time t iff t falls inside its span. -/
@[reducible] def Span.liveAt (s : Span) (t : Step) : Prop :=
  s.start ≤ t ∧ t < s.finish

/-- Length of the span (number of live steps). -/
@[reducible] def Span.length (s : Span) : ℕ := s.finish - s.start

/-- Construct the *tight* span for a buffer created at `c`, last used at `u`. -/
@[reducible] def Span.tight (created lastUsed : Step) : Span :=
  { start := created, finish := lastUsed + 1 }

theorem Span.tight_start (c u : Step) : (Span.tight c u).start  = c     := rfl
theorem Span.tight_finish (c u : Step) : (Span.tight c u).finish = u + 1 := rfl
theorem Span.tight_valid (c u : Step) (h : c ≤ u) : (Span.tight c u).valid :=
  Nat.le_succ_of_le h

/-- Two spans *overlap* if they share at least one live step. -/
@[reducible] def Span.overlaps (a b : Span) : Prop :=
  ∃ t, a.liveAt t ∧ b.liveAt t

/-!
## §6  Abstract GPU Operation

A GPU operation reads some buffers (inputs) and writes one new buffer (output).
It is scheduled on a specific stream and assigned a step number.

For a schedule to be correct:
- The writer's step must precede the reader's step on any execution path.
- Two operations on the *same* stream must have distinct steps (serial execution).
- Two operations on *different* streams with step `s₁ = s₂` may execute concurrently
  (no implicit synchronisation, unless a barrier is inserted).
-/

/-- An abstract GPU kernel or PCIe transfer. -/
structure Op (Input Output : Type) where
  stream   : StreamId  -- which CUDA stream
  execStep : Step      -- assigned time slot
  fn       : Input → Output  -- pure mathematical semantics (no side effects)

/-- A dependency edge: `op1` must complete before `op2` reads its output. -/
structure Dep (α β γ : Type) where
  producer : Op α β
  consumer : Op β γ
  /-- The producer completes before the consumer starts. -/
  h_order  : producer.execStep < consumer.execStep

end SureGPU
