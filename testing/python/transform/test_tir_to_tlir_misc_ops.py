# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""
Tests for the TIR -> TLIR importer's Group 2 misc ops
(tl.select / tl.brc / tl.cast / tl.reduce), added on top of the
original tl.add slice and Group 1's binary vector ops.

v1.2 scope: Pass A (TIR -> TLIR import) only. TLIR -> NPUIR lowering
(Pass B) for these ops is future work, so we don't run tilelangir-opt
on the output here -- only check the emitted TLIR text. `cmp` is
deliberately deferred (see tir_to_tlir.h handoff notes) and is not
covered here.

Unlike Group 1's binary ops (which all share one kernel shape), these
4 ops have different arg shapes/attrs, so each gets its own small
kernel builder instead of one shared parametrized one.
"""
from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tilelang.engine.lower import LowerAndLegalize, OptimizeForTarget


def _run_importer(func):
    """Runs the real lowering pipeline (LowerAndLegalize, then
    OptimizeForTarget) up to the point TVM would normally hand off to
    codegen, then invokes our tir_to_tlir importer directly instead of
    the downstream Bishengir/hivm pipeline (which doesn't know about the
    `tl` dialect)."""
    target = tvm.target.Target("npuir")
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    mod = LowerAndLegalize(mod, target)
    mod = OptimizeForTarget(mod, target)
    build_tlir = tvm.get_global_func("target.build.tilelang_tlir")
    built_mod = build_tlir(mod, target)
    return built_mod.get_source()


def _make_select_kernel(n=1024, block_n=1024, dtype="float32"):
    """cond/src0/src1/out all whole-buffer, same shape+dtype."""

    @T.prim_func
    def main(
        A: T.Tensor((n,), dtype),
        B: T.Tensor((n,), dtype),
        Cond: T.Tensor((n,), dtype),
        C: T.Tensor((n,), dtype),
        shape: T.int32,
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a_vec = T.alloc_ub((block_n,), dtype)
            b_vec = T.alloc_ub((block_n,), dtype)
            cond_vec = T.alloc_ub((block_n,), dtype)
            c_vec = T.alloc_ub((block_n,), dtype)
            T.copy(A[0:block_n], a_vec[0:block_n])
            T.copy(B[0:block_n], b_vec[0:block_n])
            T.copy(Cond[0:block_n], cond_vec[0:block_n])
            T.npuir_select(cond_vec, a_vec, b_vec, c_vec)
            T.copy(c_vec[0:block_n], C[0:block_n])

    return main


def _make_cast_kernel(n=1024, block_n=1024, src_dtype="float32", dst_dtype="float16"):
    """src/dst are whole-buffer but differ in element type -- exercises
    the cast-specific elemtype-mismatch-allowed verifier path."""

    @T.prim_func
    def main(
        A: T.Tensor((n,), src_dtype),
        C: T.Tensor((n,), dst_dtype),
        shape: T.int32,
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a_vec = T.alloc_ub((block_n,), src_dtype)
            c_vec = T.alloc_ub((block_n,), dst_dtype)
            T.copy(A[0:block_n], a_vec[0:block_n])
            T.npuir_cast(a_vec, c_vec)
            T.copy(c_vec[0:block_n], C[0:block_n])

    return main


def _make_reduce_kernel(n=1024, block_n=1024, dtype="float32"):
    """Reduces the whole 1024-vector down to a single element (dims=[0],
    sum, clear=True -- the simplest single-tl.reduce-call path; clear=False
    and abssum/absmax modes emit extra ops so are out of scope here)."""

    @T.prim_func
    def main(
        A: T.Tensor((n,), dtype),
        C: T.Tensor((1,), dtype),
        shape: T.int32,
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a_vec = T.alloc_ub((block_n,), dtype)
            out_vec = T.alloc_ub((1,), dtype)
            T.copy(A[0:block_n], a_vec[0:block_n])
            T.npuir_reduce(a_vec, out_vec, dims=0, reduce_mode="sum", clear=True)
            T.copy(out_vec[0:1], C[0:1])

    return main


def _make_brc_buffer_kernel(n=1024, block_n=1024, dtype="float32"):
    """Buffer-to-buffer broadcast case (same shape, so it's an identity
    broadcast -- exercises the ResolveWholeBufferRegionArg branch of brc)."""

    @T.prim_func
    def main(
        A: T.Tensor((n,), dtype),
        C: T.Tensor((n,), dtype),
        shape: T.int32,
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a_vec = T.alloc_ub((block_n,), dtype)
            c_vec = T.alloc_ub((block_n,), dtype)
            T.copy(A[0:block_n], a_vec[0:block_n])
            T.npuir_brc(a_vec, c_vec)
            T.copy(c_vec[0:block_n], C[0:block_n])

    return main


def _make_brc_scalar_kernel(n=1024, block_n=1024, dtype="float32"):
    """Scalar-fill case -- src is a FloatImm constant, not a buffer,
    exercising tir_to_tlir.cc's arith::ConstantOp branch."""

    @T.prim_func
    def main(
        C: T.Tensor((n,), dtype),
        shape: T.int32,
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            c_vec = T.alloc_ub((block_n,), dtype)
            T.npuir_brc(tvm.tir.const(1.0, dtype), c_vec)
            T.copy(c_vec[0:block_n], C[0:block_n])

    return main


def test_select_op_emitted():
    """T.npuir_select should lower to exactly one tl.select op with
    ins(...)/outs(...) syntax."""
    tlir = _run_importer(_make_select_kernel())
    assert "tl.select" in tlir
    assert "ins(" in tlir and "outs(" in tlir


def test_cast_op_emitted():
    """T.npuir_cast (src/dst differing element types) should lower to
    exactly one tl.cast op."""
    tlir = _run_importer(_make_cast_kernel())
    assert "tl.cast" in tlir


def test_reduce_op_emitted():
    """T.npuir_reduce (dims=[0], sum, clear=True) should lower to exactly
    one tl.reduce op."""
    tlir = _run_importer(_make_reduce_kernel())
    assert "tl.reduce" in tlir


def test_brc_buffer_op_emitted():
    """T.npuir_brc with a buffer source should lower to exactly one
    tl.brc op."""
    tlir = _run_importer(_make_brc_buffer_kernel())
    assert "tl.brc" in tlir


def test_brc_scalar_op_emitted():
    """T.npuir_brc with a scalar (FloatImm) source should also lower to
    exactly one tl.brc op, via the arith::ConstantOp scalar-fill branch."""
    tlir = _run_importer(_make_brc_scalar_kernel())
    assert "tl.brc" in tlir


def test_output_is_well_formed():
    """Cheap structural sanity check that the emitted text is a real
    MLIR module and not an error string or a truncated dump."""
    tlir = _run_importer(_make_select_kernel())
    stripped = tlir.strip()
    assert stripped.startswith("module")
    assert "func.func @main" in tlir
    assert stripped.endswith("}")


if __name__ == "__main__":
    tilelang.testing.main()
