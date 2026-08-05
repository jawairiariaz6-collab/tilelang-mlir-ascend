# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""
Tests for the TIR -> TLIR importer's Group 1 binary vector ops
(tl.sub / tl.mul / tl.div / tl.max / tl.min), added on top of the
original tl.add slice.

v1.1 scope: Pass A (TIR -> TLIR import) only. These ops accept memref
or tensor operands (TLIR_AnyBuffer) but this test exercises the
memref path via the same whole-buffer-region kernel shape as
test_tir_to_tlir_add.py. TLIR -> NPUIR lowering (Pass B) for these
ops is future work, so we don't run tilelangir-opt on the output
here -- only check the emitted TLIR text.
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


def _make_vec_binary_kernel(op_call, n=1024, block_n=1024, dtype="float32"):
    """Same shape as the v1 add-kernel, but calls op_call(a_vec, b_vec,
    c_vec) instead of T.vadd."""

    @T.prim_func
    def main(
        A: T.Tensor((n,), dtype),
        B: T.Tensor((n,), dtype),
        C: T.Tensor((n,), dtype),
        shape: T.int32,
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a_vec = T.alloc_ub((block_n,), dtype)
            b_vec = T.alloc_ub((block_n,), dtype)
            c_vec = T.alloc_ub((block_n,), dtype)
            T.copy(A[0:block_n], a_vec[0:block_n])
            T.copy(B[0:block_n], b_vec[0:block_n])
            op_call(a_vec, b_vec, c_vec)
            T.copy(c_vec[0:block_n], C[0:block_n])

    return main


def test_sub_op_emitted():
    """T.vsub should lower to exactly one tl.sub op with ins(...)/outs(...)."""
    tlir = _run_importer(_make_vec_binary_kernel(T.vsub))
    assert "tl.sub" in tlir
    assert "ins(" in tlir and "outs(" in tlir


def test_mul_op_emitted():
    """T.vmul should lower to exactly one tl.mul op."""
    tlir = _run_importer(_make_vec_binary_kernel(T.vmul))
    assert "tl.mul" in tlir


def test_div_op_emitted():
    """T.vdiv should lower to exactly one tl.div op."""
    tlir = _run_importer(_make_vec_binary_kernel(T.vdiv))
    assert "tl.div" in tlir


def test_max_op_emitted():
    """T.vmax should lower to exactly one tl.max op."""
    tlir = _run_importer(_make_vec_binary_kernel(T.vmax))
    assert "tl.max" in tlir


def test_min_op_emitted():
    """T.vmin should lower to exactly one tl.min op."""
    tlir = _run_importer(_make_vec_binary_kernel(T.vmin))
    assert "tl.min" in tlir


def test_output_is_well_formed():
    """Cheap structural sanity check that the emitted text is a real
    MLIR module and not an error string or a truncated dump."""
    tlir = _run_importer(_make_vec_binary_kernel(T.vsub))
    stripped = tlir.strip()
    assert stripped.startswith("module")
    assert "func.func @main" in tlir
    assert stripped.endswith("}")


if __name__ == "__main__":
    tilelang.testing.main()
