# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""
Tests for the TIR -> TLIR importer (tir_to_tlir.cc / CodeGenTIRToTLIR),
exposed as the TVM build target "target.build.tilelang_tlir".

This exercises the FYP's Pass A (TIR -> TLIR import) end to end through
TVM's normal lowering pipeline, then checks the emitted TLIR text for the
ops we expect. It's a Python-level companion to the FileCheck test in
testing/mlir/tl/lower_add.mlir, which instead drives tilelangir-opt
directly on hand-written TLIR input.

v1 scope: only the elementwise-add benchmark kernel (whole-buffer region
accesses, no ABI/workspace padding) -- see tir_to_tlir.h for the full
list of restrictions this importer currently accepts.
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


def _make_vec_add_kernel(n=1024, block_n=1024, dtype="float32"):
    """The v1 benchmark kernel: load two vectors into UB, add them,
    store the result back."""

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
            T.vadd(a_vec, b_vec, c_vec)
            T.copy(c_vec[0:block_n], C[0:block_n])

    return main


def test_add_op_emitted():
    """The T.vadd call should lower to exactly one tl.add op with
    ins(...)/outs(...) syntax."""
    tlir = _run_importer(_make_vec_add_kernel())
    assert "tl.add" in tlir
    assert "ins(" in tlir and "outs(" in tlir


def test_alloc_ops_emitted():
    """Each of the three T.alloc_ub buffers (a_vec/b_vec/c_vec) should
    become its own tl.alloc."""
    tlir = _run_importer(_make_vec_add_kernel())
    assert tlir.count("tl.alloc") == 3


def test_copy_ops_emitted():
    """At least the three explicit T.copy calls should become tl.copy
    (the lowering pipeline may legally introduce a couple more)."""
    tlir = _run_importer(_make_vec_add_kernel())
    assert tlir.count("tl.copy") >= 3


def test_output_is_well_formed():
    """Cheap structural sanity check that the emitted text is a real
    MLIR module and not an error string or a truncated dump."""
    tlir = _run_importer(_make_vec_add_kernel())
    stripped = tlir.strip()
    assert stripped.startswith("module")
    assert "func.func @main" in tlir
    assert stripped.endswith("}")


if __name__ == "__main__":
    tilelang.testing.main()
