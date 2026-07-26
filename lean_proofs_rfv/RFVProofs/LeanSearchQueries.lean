-- RFVProofs/LeanSearchQueries.lean
-- NOT part of the proven library, NOT imported by Exploration.lean either.
-- LeanSearchClient's `#leansearch` command makes an HTTP request to
-- https://leansearch.net at elaboration time, so this file is kept fully
-- separate: a flaky/offline network must not block the rest of the
-- exploration sweep. explore.sh runs this file in isolation and tolerates
-- a nonzero exit code.
--
-- Each query is a natural-language description of a lemma we suspect
-- exists in Mathlib but whose exact name we don't know — exactly the
-- "search what theorem you will need" step the user asked for.

import LeanSearchClient

-- NOTE: each query MUST end in `.` or `?` — without it, `#leansearch`
-- silently no-ops (just prints a style warning, never sends the request).
-- Found by direct testing: the original queries here were missing the
-- trailing punctuation and were never actually being sent.
#leansearch "norm of the integral of a function is at most the integral of the norm?"
#leansearch "fundamental theorem of calculus for the interval integral of the derivative?"
#leansearch "interval integral of a constant function?"
#leansearch "norm of a scalar multiple equals absolute value times norm?"
#leansearch "triangle inequality for norm of a difference of differences?"
#leansearch "absolute value of inverse of a real number?"
