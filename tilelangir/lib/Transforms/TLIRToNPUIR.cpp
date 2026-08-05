// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tilelangir/lib/Transforms/TLIRToNPUIR.cpp
 * \brief FYP Pass B: lowers TLIR (`tl` dialect) to the A5 NPUIR target
 * dialects (linalg/memref), completing the pipeline started by
 * tir_to_tlir.cc (Pass A: TIR -> TLIR).
 *
 * v1 scope, matching TLIROps.td's restrictions:
 *   tl.alloc -> memref.alloc
 *   tl.copy  -> memref.copy
 *   tl.add   -> linalg.elemwise_binary { fun = #linalg.binary_fn<add> }
 *
 * tl.add's lowering deliberately mirrors CreateLinalgBinary in
 * codegen_npuir_api_a5.cc (same builder call shape: TypeRange{}, ins,
 * outs, "fun" attr) instead of guessing a builder overload -- this is
 * the exact working pattern the real A5 codegen already uses to build
 * linalg::ElemwiseBinaryOp, so it avoids the named-structured-op builder
 * issue noted as a known limitation on the TIR -> TLIR import PR.
 */

#include "tilelangir/Transforms/Passes.h"
#include "tilelangir/Dialect/TL/TLIROps.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace tilelangir {

#define GEN_PASS_DEF_TILELANGIRTONPUIR
#include "tilelangir/Transforms/Passes.h.inc"

namespace {
#define DEBUG_TYPE "tilelangir-to-npuir"

// tl.add ins(%lhs, %rhs) outs(%out) : ...
//   -> linalg.elemwise_binary { fun = #linalg.binary_fn<add> }
//        ins(%lhs, %rhs : ...) outs(%out : ...)
struct LowerAddOp : public OpRewritePattern<mlir::tlir::AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::tlir::AddOp op,
                                 PatternRewriter &rewriter) const override {
    auto fnAttr = rewriter.getAttr<mlir::linalg::BinaryFnAttr>(
        mlir::linalg::BinaryFn::add);
    rewriter.create<mlir::linalg::ElemwiseBinaryOp>(
        op.getLoc(), mlir::TypeRange{},
        mlir::ValueRange{op.getLhs(), op.getRhs()},
        mlir::ValueRange{op.getOut()},
        mlir::ArrayRef{rewriter.getNamedAttr("fun", fnAttr)});
    rewriter.eraseOp(op);
    return success();
  }
};

// tl.copy %source -> %target : ...  ->  memref.copy %source, %target
struct LowerCopyOp : public OpRewritePattern<mlir::tlir::CopyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::tlir::CopyOp op,
                                 PatternRewriter &rewriter) const override {
    rewriter.create<mlir::memref::CopyOp>(op.getLoc(), op.getSource(),
                                           op.getTarget());
    rewriter.eraseOp(op);
    return success();
  }
};

// %result = tl.alloc : memref<...>  ->  %result = memref.alloc() : memref<...>
struct LowerAllocOp : public OpRewritePattern<mlir::tlir::AllocOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::tlir::AllocOp op,
                                 PatternRewriter &rewriter) const override {
    auto memrefTy = llvm::cast<MemRefType>(op.getResult().getType());
    auto newAlloc =
        rewriter.create<mlir::memref::AllocOp>(op.getLoc(), memrefTy);
    rewriter.replaceOp(op, newAlloc.getResult());
    return success();
  }
};

struct TileLangIRToNPUIR : impl::TileLangIRToNPUIRBase<TileLangIRToNPUIR> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerAddOp, LowerCopyOp, LowerAllocOp>(&getContext());
    // Each pattern only matches on its own tl.* op kind; tl.copy/tl.add
    // consume a tl.alloc result by SSA value, but the greedy driver
    // revisits users after a replacement, so lowering order across the
    // three patterns doesn't need to be forced explicitly here.
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[" DEBUG_TYPE "]: pattern application did not converge\n");
      signalPassFailure();
    }
  }
};
#undef DEBUG_TYPE
} // namespace

} // namespace tilelangir
} // namespace mlir
