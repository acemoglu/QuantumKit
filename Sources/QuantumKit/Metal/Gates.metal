//
//  File.swift
//  QuantumKit
//
//  Created by Bugra Acemoglu on 13.06.2026.
//

#include <metal_stdlib>
using namespace metal;

kernel void hadamard_gate(device float* realBuffer [[buffer(0)]],
                          device float* imagBuffer [[buffer(1)]],
                          constant uint& targetQubit [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {

    uint mask = (1 << targetQubit) - 1;
    uint i0 = ((id >> targetQubit) << (targetQubit + 1)) | (id & mask);
    uint i1 = i0 | (1 << targetQubit);
    
    // 2. Taking data from RAM (Unified Memory)
    float r0 = realBuffer[i0];
    float im0 = imagBuffer[i0];
    
    float r1 = realBuffer[i1];
    float im1 = imagBuffer[i1];

    // Multiplication without matrices (for performance optimization)
    // Using Metal's built in M_SQRT1_2_F (1/sqrt(2)) for maximum Float32 precision
    
    realBuffer[i0] = (r0 + r1) * M_SQRT1_2_F;
    imagBuffer[i0] = (im0 + im1) * M_SQRT1_2_F;
    
    realBuffer[i1] = (r0 - r1) * M_SQRT1_2_F;
    imagBuffer[i1] = (im0 - im1) * M_SQRT1_2_F;
}
