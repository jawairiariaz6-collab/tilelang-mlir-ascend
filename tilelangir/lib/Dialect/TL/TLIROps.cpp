// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include "tilelangir/Dialect/TL/TLIROps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::tlir;

LogicalResult CopyOp::verify() {
  auto srcTy = llvm::cast<MemRefType>(getSource().getType());
  auto dstTy = llvm::cast<MemRefType>(getTarget().getType());
  if (srcTy.getShape() != dstTy.getShape())
    return emitOpError("source and target shapes must match, got ")
           << srcTy << " and " << dstTy;
  if (srcTy.getElementType() != dstTy.getElementType())
    return emitOpError("source and target element types must match");
  return success();
}

LogicalResult AddOp::verify() {
  auto lhsTy = llvm::cast<MemRefType>(getLhs().getType());
  auto rhsTy = llvm::cast<MemRefType>(getRhs().getType());
  auto outTy = llvm::cast<MemRefType>(getOut().getType());

  if (lhsTy.getShape() != rhsTy.getShape() ||
      lhsTy.getShape() != outTy.getShape())
    return emitOpError(
        "lhs, rhs and out must have identical shapes (v1: no broadcast; "
        "extend before adding a kernel that needs it)");

  if (lhsTy.getElementType() != rhsTy.getElementType() ||
      lhsTy.getElementType() != outTy.getElementType())
    return emitOpError("lhs, rhs and out must have identical element type");

  if (!lhsTy.getElementType().isIntOrFloat())
    return emitOpError("tl.add currently only supports int/float element "
                        "types");

  return success();
}

// Shared verifier logic for the binary vector ops (sub/mul/div/max/min) --
// mirrors AddOp::verify() but works for memref OR tensor operands via
// ShapedType (the common base class both inherit from).
static LogicalResult verifyBinaryVecOp(Operation *op, Value lhs, Value rhs,
                                        Value out) {
  auto lhsTy = llvm::cast<ShapedType>(lhs.getType());
  auto rhsTy = llvm::cast<ShapedType>(rhs.getType());
  auto outTy = llvm::cast<ShapedType>(out.getType());

  if (lhsTy.getShape() != rhsTy.getShape() ||
      lhsTy.getShape() != outTy.getShape())
    return op->emitOpError("lhs, rhs and out must have identical shapes");

  if (lhsTy.getElementType() != rhsTy.getElementType() ||
      lhsTy.getElementType() != outTy.getElementType())
    return op->emitOpError("lhs, rhs and out must have identical element type");

  if (!lhsTy.getElementType().isIntOrFloat())
    return op->emitOpError("only supports int/float element types");

  return success();
}

LogicalResult SubOp::verify() {
  return verifyBinaryVecOp(getOperation(), getLhs(), getRhs(), getOut());
}
LogicalResult MulOp::verify() {
  return verifyBinaryVecOp(getOperation(), getLhs(), getRhs(), getOut());
}
LogicalResult DivOp::verify() {
  return verifyBinaryVecOp(getOperation(), getLhs(), getRhs(), getOut());
}
LogicalResult MaxOp::verify() {
  return verifyBinaryVecOp(getOperation(), getLhs(), getRhs(), getOut());
}
LogicalResult MinOp::verify() {
  return verifyBinaryVecOp(getOperation(), getLhs(), getRhs(), getOut());
}

// v1.2: select/brc/cast/reduce -- each op gets its own verifier since
// shapes/attrs differ (see TLIROps.td comment above the definitions).

LogicalResult SelectOp::verify() {
  auto condTy = llvm::cast<ShapedType>(getCond().getType());
  auto src0Ty = llvm::cast<ShapedType>(getSrc0().getType());
  auto src1Ty = llvm::cast<ShapedType>(getSrc1().getType());
  auto outTy = llvm::cast<ShapedType>(getOut().getType());

  if (condTy.getShape() != src0Ty.getShape())
    return emitOpError("cond and src0/src1/out must have identical shapes");

  if (src0Ty.getShape() != src1Ty.getShape() ||
      src0Ty.getShape() != outTy.getShape())
    return emitOpError("src0, src1 and out must have identical shapes");

  if (src0Ty.getElementType() != src1Ty.getElementType() ||
      src0Ty.getElementType() != outTy.getElementType())
    return emitOpError("src0, src1 and out must have identical element type");

  if (!src0Ty.getElementType().isIntOrFloat())
    return emitOpError("only supports int/float element types for src0/src1/out");

  return success();
}

LogicalResult BrcOp::verify() {
  auto outTy = llvm::cast<ShapedType>(getOut().getType());

  // Scalar-fill case: `src` is a plain int/float value, not a buffer.
  if (!llvm::isa<ShapedType>(getSrc().getType())) {
    if (!getSrc().getType().isIntOrFloat())
      return emitOpError("scalar src must be int or float typed");
    return success();
  }

  // Buffer-to-buffer broadcast case: shapes must be broadcast-compatible
  // (each src dim must equal the matching out dim, or be 1).
  auto srcTy = llvm::cast<ShapedType>(getSrc().getType());
  if (srcTy.getElementType() != outTy.getElementType())
    return emitOpError("src and out must have identical element type");

  auto srcShape = srcTy.getShape();
  auto outShape = outTy.getShape();
  if (srcShape.size() != outShape.size())
    return emitOpError("src and out must have identical rank");

  for (size_t i = 0; i < srcShape.size(); ++i) {
    if (srcShape[i] != outShape[i] && srcShape[i] != 1)
      return emitOpError("src shape must be broadcastable to out shape, got ")
             << srcTy << " and " << outTy;
  }

  return success();
}

LogicalResult CastOp::verify() {
  auto srcTy = llvm::cast<ShapedType>(getSrc().getType());
  auto outTy = llvm::cast<ShapedType>(getOut().getType());

  // Element types are intentionally allowed to differ -- that is the
  // purpose of this op. Only check both are int/float (no casts to/from
  // e.g. index or other non-numeric types).
  if (!srcTy.getElementType().isIntOrFloat() ||
      !outTy.getElementType().isIntOrFloat())
    return emitOpError("src and out element types must be int or float");

  auto srcShape = srcTy.getShape();
  auto outShape = outTy.getShape();
  if (srcShape.size() != outShape.size())
    return emitOpError("src and out must have identical rank");

  for (size_t i = 0; i < srcShape.size(); ++i) {
    if (srcShape[i] != outShape[i] && srcShape[i] != 1)
      return emitOpError("src shape must be broadcastable to out shape, got ")
             << srcTy << " and " << outTy;
  }

  return success();
}

LogicalResult ReduceOp::verify() {
  auto srcTy = llvm::cast<ShapedType>(getSrc().getType());
  auto outTy = llvm::cast<ShapedType>(getOut().getType());

  if (srcTy.getElementType() != outTy.getElementType())
    return emitOpError("src and out must have identical element type");

  if (!srcTy.getElementType().isIntOrFloat())
    return emitOpError("only supports int/float element types");

  auto srcShape = srcTy.getShape();
  auto outShape = outTy.getShape();
  if (srcShape.size() != outShape.size())
    return emitOpError(
        "src and out must have identical rank (reduced axes keep size 1 "
        "in out, they are not squeezed)");

  ArrayRef<int64_t> reduceDims = getReduceDims();
  if (reduceDims.empty())
    return emitOpError("reduce_dims must not be empty");

  for (int64_t dim : reduceDims) {
    if (dim < 0 || static_cast<size_t>(dim) >= srcShape.size())
      return emitOpError("reduce_dims entry ")
             << dim << " is out of bounds for rank " << srcShape.size();
  }

  auto isReducedDim = [&](int64_t i) {
    for (int64_t d : reduceDims)
      if (d == i)
        return true;
    return false;
  };

  for (size_t i = 0; i < srcShape.size(); ++i) {
    if (isReducedDim(static_cast<int64_t>(i))) {
      if (outShape[i] != 1)
        return emitOpError("out shape must have size 1 along reduced dim ")
               << i << ", got " << outTy;
    } else {
      if (outShape[i] != srcShape[i])
        return emitOpError("out shape must match src shape along "
                           "non-reduced dim ")
               << i << ", got " << srcTy << " and " << outTy;
    }
  }

  return success();
}

