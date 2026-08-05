// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
/*!
 * \file tilelangir/include/tilelangir/Transforms/Passes.td
 * \brief TileLangIR pass TableGen definitions (CV split, vectorize, TLIR->NPUIR).
 *
 */
#ifndef TILELANGIR_TRANSFORMS_PASSES_TD
#define TILELANGIR_TRANSFORMS_PASSES_TD
include "mlir/Pass/PassBase.td"
def TileLangIRCVSplit : Pass<"tilelangir-cv-split", "::mlir::ModuleOp"> {
  let summary = "TileLangIR CV split pass (custom)";
  let description = [{
    Splits or transforms operations for CV (e.g. coefficient/variable separation).
    Implementation to be added.
  }];
}
def TileLangIRVectorize : Pass<"tilelangir-vectorize", "::mlir::ModuleOp"> {
  let summary = "TileLangIR vectorize pass (custom)";
  let description = [{
    Applies vectorization transformations. Implementation to be added.
  }];
}

def TileLangIRToNPUIR : Pass<"tilelangir-to-npuir", "::mlir::ModuleOp"> {
  let summary = "Lower TLIR (`tl` dialect) ops to the A5 NPUIR target dialects";
  let description = [{
    FYP Pass B: completes the TIR -> TLIR -> NPUIR pipeline for the v1
    (add-kernel) TLIR op set, following the codegen_npuir_api_a5.cc target
    (linalg/memref), not the A2/A3 hivm::VAddOp path:
      tl.alloc -> memref.alloc
      tl.copy  -> memref.copy
      tl.add   -> linalg.elemwise_binary { fun = #linalg.binary_fn<add> }
  }];
  let dependentDialects = [
    "::mlir::linalg::LinalgDialect",
    "::mlir::memref::MemRefDialect"
  ];
}

#endif // TILELANGIR_TRANSFORMS_PASSES_TD
