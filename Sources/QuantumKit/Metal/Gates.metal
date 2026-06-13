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

kernel void pauli_x_gate(device float* realBuffer [[buffer(0)]],
                         device float* imagBuffer [[buffer(1)]],
                         constant uint& targetQubit [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {

    uint mask = (1 << targetQubit) - 1;
    uint i0 = ((id >> targetQubit) << (targetQubit + 1)) | (id & mask);
    uint i1 = i0 | (1 << targetQubit);

    // Swap amplitudes completely
    float tempR = realBuffer[i0];
    float tempI = imagBuffer[i0];

    realBuffer[i0] = realBuffer[i1];
    imagBuffer[i0] = imagBuffer[i1];

    realBuffer[i1] = tempR;
    imagBuffer[i1] = tempI;
}

kernel void pauli_y_gate(device float* realBuffer [[buffer(0)]],
                         device float* imagBuffer [[buffer(1)]],
                         constant uint& targetQubit [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {

    uint mask = (1 << targetQubit) - 1;
    uint i0 = ((id >> targetQubit) << (targetQubit + 1)) | (id & mask);
    uint i1 = i0 | (1 << targetQubit);

    float r0 = realBuffer[i0];
    float im0 = imagBuffer[i0];
    
    float r1 = realBuffer[i1];
    float im1 = imagBuffer[i1];

    // Y gate applies: |0> -> i|1>, |1> -> -i|0>
    // Complex multiplication rules dictate this specific cross-assignment
    realBuffer[i0] = im1;
    imagBuffer[i0] = -r1;

    realBuffer[i1] = -im0;
    imagBuffer[i1] = r0;
}


kernel void pauli_z_gate(device float* realBuffer [[buffer(0)]],
                         device float* imagBuffer [[buffer(1)]],
                         constant uint& targetQubit [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {

    uint mask = (1 << targetQubit) - 1;
    uint i0 = ((id >> targetQubit) << (targetQubit + 1)) | (id & mask);
    uint i1 = i0 | (1 << targetQubit);

    // State |0> (index i0) remains unchanged.
    // State |1> (index i1) flips its phase (-1).
    realBuffer[i1] = -realBuffer[i1];
    imagBuffer[i1] = -imagBuffer[i1];
}

// Note: Buffer 2  expects a uint2 (Control and Target qubits)
kernel void cnot_gate(device float* realBuffer [[buffer(0)]],
                      device float* imagBuffer [[buffer(1)]],
                      constant uint2& qubits [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {

    uint controlQubit = qubits.x;
    uint targetQubit = qubits.y;

    // Base parallel split on the target qubit
    uint mask = (1 << targetQubit) - 1;
    uint i0 = ((id >> targetQubit) << (targetQubit + 1)) | (id & mask);
    uint i1 = i0 | (1 << targetQubit);

    // Apply the Pauli-X swap only if the control bit is 1. Bitwise check on the state index ensures exact entanglement
    if ((i0 & (1 << controlQubit)) != 0) {
        float tempR = realBuffer[i0];
        float tempI = imagBuffer[i0];

        realBuffer[i0] = realBuffer[i1];
        imagBuffer[i0] = imagBuffer[i1];

        realBuffer[i1] = tempR;
        imagBuffer[i1] = tempI;
    }
}

// RZ: |0> *= e^{-i theta/2}, |1> *= e^{i theta/2}
kernel void rz_gate(device float* realBuffer [[buffer(0)]],
                    device float* imagBuffer [[buffer(1)]],
                    constant uint& targetQubit [[buffer(2)]],
                    constant float& theta [[buffer(3)]],
                    uint id [[thread_position_in_grid]]) {

    uint mask = (1 << targetQubit) - 1;
    uint i0 = ((id >> targetQubit) << (targetQubit + 1)) | (id & mask);
    uint i1 = i0 | (1 << targetQubit);

    float r0 = realBuffer[i0];
    float im0 = imagBuffer[i0];

    float r1 = realBuffer[i1];
    float im1 = imagBuffer[i1];

    float halfTheta = theta * 0.5f;
    float cosH = cos(halfTheta);
    float sinH = sin(halfTheta);

    // i0 multiplied by (cosH - i sinH)
    realBuffer[i0] = r0 * cosH + im0 * sinH;
    imagBuffer[i0] = im0 * cosH - r0 * sinH;

    // i1 multiplied by (cosH + i sinH)
    realBuffer[i1] = r1 * cosH - im1 * sinH;
    imagBuffer[i1] = im1 * cosH + r1 * sinH;
}

// Note: Buffer 2 expects a uint3 (control1, control2, target)
kernel void ccx_gate(device float* realBuffer [[buffer(0)]],
                     device float* imagBuffer [[buffer(1)]],
                     constant uint3& qubits [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {

    uint control1 = qubits.x;
    uint control2 = qubits.y;
    uint targetQubit = qubits.z;

    uint mask = (1 << targetQubit) - 1;
    uint i0 = ((id >> targetQubit) << (targetQubit + 1)) | (id & mask);
    uint i1 = i0 | (1 << targetQubit);

    if ((i0 & (1 << control1)) != 0 && (i0 & (1 << control2)) != 0) {
        float tempR = realBuffer[i0];
        float tempI = imagBuffer[i0];

        realBuffer[i0] = realBuffer[i1];
        imagBuffer[i0] = imagBuffer[i1];

        realBuffer[i1] = tempR;
        imagBuffer[i1] = tempI;
    }
}

// Measurement (Probability Calculation)
kernel void compute_probabilities(device float* realBuffer [[buffer(0)]],
                                  device float* imagBuffer [[buffer(1)]],
                                  device float* probBuffer [[buffer(2)]],
                                  uint id [[thread_position_in_grid]]) {
                                      
    float r = realBuffer[id];
    float i = imagBuffer[id];
    
    // Born rule: P(A) = |amplitude|^2
    probBuffer[id] = (r * r) + (i * i);
}

constant uint scanBlockSize [[function_constant(0)]];

inline uint prefix_scan_block_size() {
    return scanBlockSize > 0 ? scanBlockSize : 256;
}

inline void threadgroup_inclusive_scan(threadgroup float* sharedData, uint localID, uint groupSize) {
    for (uint stride = 1; stride < groupSize; stride <<= 1) {
        float accumulated = 0.0f;
        if (localID >= stride) {
            accumulated = sharedData[localID - stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sharedData[localID] += accumulated;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Phase 0: block-local inclusive scan on `dataBuffer`, storing each block total in `blockSums`.
// Phase 2: add exclusive block offsets from scanned `blockSums` into `dataBuffer`.
kernel void prefix_sum_probabilities(device float* dataBuffer [[buffer(0)]],
                                     device float* blockSums [[buffer(1)]],
                                     constant uint& elementCount [[buffer(2)]],
                                     constant uint& phase [[buffer(3)]],
                                     uint globalID [[thread_position_in_grid]],
                                     uint localID [[thread_index_in_threadgroup]],
                                     uint groupID [[threadgroup_position_in_grid]],
                                     uint groupSize [[threads_per_threadgroup]]) {
    const uint blockSize = prefix_scan_block_size();
    threadgroup float sharedData[256];

    if (phase == 0) {
        const uint index = groupID * groupSize + localID;
        const float value = (index < elementCount) ? dataBuffer[index] : 0.0f;
        sharedData[localID] = value;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup_inclusive_scan(sharedData, localID, groupSize);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (index < elementCount) {
            dataBuffer[index] = sharedData[localID];
        }

        const uint blockEnd = min((groupID + 1) * groupSize, elementCount);
        if (blockEnd > 0 && index == blockEnd - 1) {
            blockSums[groupID] = sharedData[localID];
        }
        return;
    }

    if (phase == 2) {
        const uint index = groupID * groupSize + localID;
        if (index >= elementCount) {
            return;
        }

        const float blockOffset = (groupID > 0) ? blockSums[groupID - 1] : 0.0f;
        dataBuffer[index] += blockOffset;
    }
}

kernel void find_collapsed_state(device const float* cdfBuffer [[buffer(0)]],
                                 constant float& diceRoll [[buffer(1)]],
                                 constant uint& elementCount [[buffer(2)]],
                                 device uint* collapsedIndex [[buffer(3)]],
                                 uint threadID [[thread_position_in_grid]]) {
    if (threadID != 0 || elementCount == 0) {
        return;
    }

    uint low = 0;
    uint high = elementCount - 1;
    uint result = elementCount - 1;

    while (low <= high) {
        const uint mid = (low + high) >> 1;
        if (diceRoll < cdfBuffer[mid]) {
            result = mid;
            if (mid == 0) {
                break;
            }
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }

    collapsedIndex[0] = result;
}


