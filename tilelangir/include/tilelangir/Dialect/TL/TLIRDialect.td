//===- TLIRDialect.td - TLIR dialect definition -----------*- tablegen -*-===//
#ifndef TILELANGIR_DIALECT_TL_TLIRDIALECT_TD
#define TILELANGIR_DIALECT_TL_TLIRDIALECT_TD

include "mlir/IR/OpBase.td"

def TLIR_Dialect : Dialect {
  let name = "tl";
  let summary = "Restricted TileLang IR dialect (TLIR)";
  let description = [{
    TLIR is an intermediate representation layer between TVM TIR (as
    emitted by TileLang for Ascend) and the A5 NPUIR target dialects
    (linalg for elementwise vector ops, memref for buffers/DMA).

    This restricted slice covers only the ops needed to represent and
    lower the elementwise `add` benchmark kernel:
      tl.alloc, tl.copy, tl.add
  }];
  let cppNamespace = "::mlir::tlir";
  let useDefaultAttributePrinterParser = 1;
}

class TLIR_Op<string mnemonic, list<Trait> traits = []>
    : Op<TLIR_Dialect, mnemonic, traits>;

#endif // TILELANGIR_DIALECT_TL_TLIRDIALECT_TD
