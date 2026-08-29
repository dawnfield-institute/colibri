# Verifying the qwen36 routing-telemetry wiring

Record of what was actually checked when `route_trace.h` was wired into
`c/qwen36.c`, and one thing found along the way that is not about that change.

Machine: node1, i9-12900K, Debian trixie, gcc 14.2. Engine at fork
`sync/upstream-v1.9.0` (upstream `main` @ 184e052, v1.9.0).

## The invariant that matters

Telemetry must observe and not participate. Fixture: `tools/make_qwen36_tiny.py`
(0.6M params, the real hybrid layout -- 8 layers, 2 Gated Attention + 6 Gated
DeltaNet, 8 experts, top-2), converted with `tools/convert_qwen36.py`.

| run | tokens matched | hit / miss |
|---|---|---|
| telemetry unset | 9/16 | 256 / 64 |
| `ROUTE_TRACE=... COLI_USAGE=...` | 9/16 | 256 / 64 |

Identical generated token sequence and identical cache accounting, on a real
forward pass through both layer kinds. Repeated against both reference modes
(below) with the same result. `COLI_USAGE` wrote 320 selections across 64
distinct experts and reloads.

Unit tests, all passing: `test_route_trace`, `test_798_guards`,
`test_qwen36_cache_index`, `test_qwen36_ctx`, `test_qwen36_dense_batch`,
`test_qwen36_json_escape`, plus `test_olmoe_cache_index` and
`test_olmoe_matmul_q` for the olmoe changes in the same branch.

## Not token-exact against the tiny fixture, and not because of this change

Worth writing down separately, because it is easy to read the table above as a
failed exactness gate. It is not one: the divergence is identical with the
telemetry off, so it predates and is untouched by this wiring.

`make_qwen36_tiny.py --emit-ref` defaults to `--ref-mode attention_only`, which
its own docstring describes as "matching what qwen36.c Phase 1 computes" -- it
replaces the six DeltaNet layers with identity. The engine is well past Phase 1
and computes them for real, so that reference cannot be met:

| reference | int4-gs64 | int8 |
|---|---|---|
| `attention_only` (default) | 6/16 | 6/16 |
| `full` | 9/16 | 9/16 |

`--ref-mode full` recovers three tokens, which is the direction the DeltaNet
explanation predicts -- the recurrent state starts near zero and the two
computations part company as it accumulates. It does not recover all of them,
and the remaining gap is **precision-independent**: int4 and int8 produce the
same 9/16 and the same divergent continuation, so expert quantization is ruled
out as the cause.

That is as far as the evidence goes. What it supports: the tiny fixture is not
currently usable as a token-exactness gate for this engine, and its default
reference mode is stale by a phase. What it does not support: any claim about
where the residual difference lives. A 0.6M-parameter random model under greedy
decoding is a knife edge, and the intended tolerance upstream is unknown.

Worth reporting upstream as an observation, with the control that rules out
quantization, and without a diagnosis attached.
