//===- TLIROps.td - TLIR op definitions (alloc/copy/add slice) ----------===//
#ifndef TILELANGIR_DIALECT_TL_TLIROPS_TD
#define TILELANGIR_DIALECT_TL_TLIROPS_TD

include "TLIRDialect.td"
include "TLIRAttrs.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinTypes.td"

def TLIR_AllocOp : TLIR_Op<"alloc", []> {
  let summary = "Allocate a tile buffer in a given address space";
  let description = [{
    Corresponds to `T.alloc_ub` / `T.alloc_shared` in TileLang. Lowers to
    `memref.alloc` in the TLIR -> NPUIR pass.

    Example:
      %ub = tl.alloc : memref<4096xf32, #tl.address_space<ub>>
  }];
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def TLIR_CopyOp : TLIR_Op<"copy", []> {
  let summary = "Copy between two tile buffers";
  let description = [{
    Corresponds to `T.copy(src, dst)`. v1 covers the same-shape case only.

    Example:
      tl.copy %gm_a -> %ub_a : memref<4096xf32>, memref<4096xf32>
  }];
  let arguments = (ins AnyMemRef:$source, AnyMemRef:$target);
  let assemblyFormat = "$source `->` $target attr-dict `:` type($source) `,` type($target)";
  let hasVerifier = 1;
}

def TLIR_AddOp : TLIR_Op<"add", []> {
  let summary = "Elementwise tile addition (vector core)";
  let description = [{
    Corresponds to `T.npuir_add(A_VEC, B_VEC, C_VEC)`. Lowers to
    `linalg.elemwise_binary { fun = #linalg.binary_fn<add> }` (A5 target).

    Example:
      tl.add ins(%ub_a, %ub_b) outs(%ub_c) : memref<4096xf32>, memref<4096xf32>, memref<4096xf32>
  }];
  let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs, AnyMemRef:$out);
  let assemblyFormat =
      "`ins` `(` $lhs `,` $rhs `)` `outs` `(` $out `)` attr-dict `:` "
      "type($lhs) `,` type($rhs) `,` type($out)";
  let hasVerifier = 1;
}

// v1.1: memref OR tensor operands, for tl.sub/mul/div/max/min and beyond --
// unlike tl.add (memref-only, kept as-is above since it already has a
// passing FileCheck test we don't want to risk), these new binary ops are
// written from the start to accept either representation.
def TLIR_AnyBuffer : AnyTypeOf<[AnyMemRef, AnyRankedTensor]>;

class TLIR_BinaryVecOp<string mnemonic, string opDescription>
    : TLIR_Op<mnemonic, []> {
  let summary = "Elementwise tile " # opDescription # " (vector core)";
  let description = [{
    Corresponds to `T.npuir_}] # mnemonic # [{(A_VEC, B_VEC, C_VEC)`. Import
    (TIR -> TLIR) only for now; TLIR -> NPUIR lowering for this op is
    future work (see TLIRToNPUIR.cpp).

    Example:
      tl.}] # mnemonic # [{ ins(%ub_a, %ub_b) outs(%ub_c) : memref<4096xf32>, memref<4096xf32>, memref<4096xf32>
  }];
  let arguments = (ins TLIR_AnyBuffer:$lhs, TLIR_AnyBuffer:$rhs, TLIR_AnyBuffer:$out);
  let assemblyFormat =
      "`ins` `(` $lhs `,` $rhs `)` `outs` `(` $out `)` attr-dict `:` "
      "type($lhs) `,` type($rhs) `,` type($out)";
  let hasVerifier = 1;
}

def TLIR_SubOp : TLIR_BinaryVecOp<"sub", "subtraction">;
def TLIR_MulOp : TLIR_BinaryVecOp<"mul", "multiplication">;
def TLIR_DivOp : TLIR_BinaryVecOp<"div", "division">;
def TLIR_MaxOp : TLIR_BinaryVecOp<"max", "maximum">;
def TLIR_MinOp : TLIR_BinaryVecOp<"min", "minimum">;

// v1.2: select/brc/cast/reduce -- heterogeneous shapes/attrs, each op
// defined individually (unlike the binary ops above, no single shared
// base class fits all four). Import (TIR -> TLIR) only for now; TLIR ->
// NPUIR lowering for these ops is future work (see TLIRToNPUIR.cpp).

def TLIR_SelectOp : TLIR_Op<"select", []> {
  let summary = "Elementwise select based on a condition mask (vector core)";
  let description = [{
    Corresponds to `T.npuir_select(Cond_VEC, A_VEC, B_VEC, C_VEC)`. `cond`
    may have a different element type than `src0`/`src1`/`out` (mask type).
    Import (TIR -> TLIR) only for now; TLIR -> NPUIR lowering for this op
    is future work (see TLIRToNPUIR.cpp).

    Example:
      tl.select ins(%cond, %ub_a, %ub_b) outs(%ub_c) : memref<32x64xi1>, memref<32x64xf16>, memref<32x64xf16>, memref<32x64xf16>
  }];
  let arguments = (ins TLIR_AnyBuffer:$cond, TLIR_AnyBuffer:$src0, TLIR_AnyBuffer:$src1, TLIR_AnyBuffer:$out);
  let assemblyFormat =
      "`ins` `(` $cond `,` $src0 `,` $src1 `)` `outs` `(` $out `)` attr-dict `:` "
      "type($cond) `,` type($src0) `,` type($src1) `,` type($out)";
  let hasVerifier = 1;
}

def TLIR_BrcOp : TLIR_Op<"brc", []> {
  let summary = "Broadcast a buffer or scalar into a destination buffer (vector core)";
  let description = [{
    Corresponds to `T.npuir_brc(src, dst)`. `src` may be either a buffer
    (buffer-to-buffer broadcast) or a scalar constant (scalar-fill case).
    Import (TIR -> TLIR) only for now; TLIR -> NPUIR lowering for this op
    is future work (see TLIRToNPUIR.cpp).

    Example:
      tl.brc ins(%ub_a) outs(%ub_c) : memref<1x64xf16>, memref<32x64xf16>
  }];
  let arguments = (ins AnyTypeOf<[TLIR_AnyBuffer, AnyFloat, AnySignlessInteger]>:$src, TLIR_AnyBuffer:$out);
  let assemblyFormat =
      "`ins` `(` $src `)` `outs` `(` $out `)` attr-dict `:` "
      "type($src) `,` type($out)";
  let hasVerifier = 1;
}

def TLIR_CastOp : TLIR_Op<"cast", []> {
  let summary = "Elementwise type conversion (vector core)";
  let description = [{
    Corresponds to `T.npuir_cast(src, dst, round_mode)`. `src`/`dst` may
    have different element types (that is the purpose of this op); shapes
    may also differ (broadcast). Import (TIR -> TLIR) only for now; TLIR ->
    NPUIR lowering for this op is future work (see TLIRToNPUIR.cpp).

    Example:
      tl.cast ins(%ub_a) outs(%ub_c) {round_mode = "rint"} : memref<32x64xf32>, memref<32x64xf16>
  }];
  let arguments = (ins TLIR_AnyBuffer:$src, TLIR_AnyBuffer:$out, StrAttr:$round_mode);
  let assemblyFormat =
      "`ins` `(` $src `)` `outs` `(` $out `)` attr-dict `:` "
      "type($src) `,` type($out)";
  let hasVerifier = 1;
}

def TLIR_ReduceOp : TLIR_Op<"reduce", []> {
  let summary = "Reduce a buffer along given axes (vector core)";
  let description = [{
    Corresponds to `T.npuir_reduce(src, dst, dims, reduce_mode)`. `src`
    and `dst` always have the same rank; reduced axes keep size 1 in
    `dst` (no squeeze). Import (TIR -> TLIR) only for now; TLIR -> NPUIR
    lowering for this op is future work (see TLIRToNPUIR.cpp).

    Example:
      tl.reduce ins(%ub_a) outs(%ub_c) {reduce_mode = "sum", reduce_dims = array<i64: 1>} : memref<32x64xf32>, memref<32x1xf32>
  }];
  let arguments = (ins TLIR_AnyBuffer:$src, TLIR_AnyBuffer:$out, StrAttr:$reduce_mode, DenseI64ArrayAttr:$reduce_dims);
  let assemblyFormat =
      "`ins` `(` $src `)` `outs` `(` $out `)` attr-dict `:` "
      "type($src) `,` type($out)";
  let hasVerifier = 1;
}

#endif // TILELANGIR_DIALECT_TL_TLIROPS_TD
