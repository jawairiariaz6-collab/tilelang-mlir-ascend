// RUN: tilelangir-opt %s --tilelangir-to-npuir | FileCheck %s

func.func @test_lower(%src: memref<1024xf32>, %dst: memref<1024xf32>) {
  %ub = tl.alloc : memref<1024xf32, #tl.address_space<ub>>
  tl.copy %src -> %dst : memref<1024xf32>, memref<1024xf32>
  tl.add ins(%src, %dst) outs(%ub) : memref<1024xf32>, memref<1024xf32>, memref<1024xf32, #tl.address_space<ub>>
  return
}

// CHECK: memref.alloc
// CHECK: memref.copy
// CHECK: linalg.elemwise_binary
// CHECK-SAME: #linalg.binary_fn<add>
