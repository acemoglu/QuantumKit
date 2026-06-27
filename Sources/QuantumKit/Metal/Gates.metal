//
//  File.swift
//  QuantumKit
//
//  Created by Bugra Acemoglu on 13.06.2026.
//

#include <metal_stdlib>
using namespace metal;

// MARK: - Double-single (compensated) arithmetic
//
// To fight Float32 catastrophic cancellation while building the cumulative distribution (the CDF
// plateaus that appear above ~24 qubits), probability sums are carried as an unevaluated pair
// `value = hi + lo`, where `lo` holds the rounding error that a single float would discard. This is
// the same idea as Kahan summation, reshaped into an associative operator so it survives a parallel
// (Hillis–Steele) prefix scan. The compensation lives entirely in thread registers — no extra VRAM
// traffic — so the gate kernels stay pure, single-precision Float32 and run at full speed.
//
// The two kernels that use these helpers (`prefix_sum_probabilities`, `find_collapsed_state`) are
// compiled into a separate library with fast math OFF, so the compiler cannot algebraically cancel
// the error term (`err`). That makes a `volatile` barrier unnecessary here and lets the scheduler
// pipeline the dependent rounding steps — important since this scan is bandwidth-bound.

inline float2 compensated_two_sum(float a, float b) {
    float s = a + b;
    float bb = s - a;
    float err = (a - (s - bb)) + (b - bb);
    return float2(s, err);
}

// Sum of two double-single numbers, renormalized so |lo| <= 0.5 ulp(hi).
inline float2 df_add(float2 a, float2 b) {
    float2 s = compensated_two_sum(a.x, b.x);
    float lo = s.y + (a.y + b.y);
    return compensated_two_sum(s.x, lo);
}

// Lexicographic comparison of double-single numbers: a < b.
inline bool df_less(float2 a, float2 b) {
    return (a.x < b.x) || (a.x == b.x && a.y < b.y);
}

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

// S: |1> *= i  (RZ(pi/2))
kernel void s_gate(device float* realBuffer [[buffer(0)]],
                   device float* imagBuffer [[buffer(1)]],
                   constant uint& targetQubit [[buffer(2)]],
                   uint id [[thread_position_in_grid]]) {

    uint mask = (1 << targetQubit) - 1;
    uint i0 = ((id >> targetQubit) << (targetQubit + 1)) | (id & mask);
    uint i1 = i0 | (1 << targetQubit);

    float r1 = realBuffer[i1];
    float im1 = imagBuffer[i1];

    realBuffer[i1] = -im1;
    imagBuffer[i1] = r1;
}

// T: |1> *= e^{i pi/4}  (RZ(pi/4))
kernel void t_gate(device float* realBuffer [[buffer(0)]],
                   device float* imagBuffer [[buffer(1)]],
                   constant uint& targetQubit [[buffer(2)]],
                   uint id [[thread_position_in_grid]]) {

    uint mask = (1 << targetQubit) - 1;
    uint i0 = ((id >> targetQubit) << (targetQubit + 1)) | (id & mask);
    uint i1 = i0 | (1 << targetQubit);

    float r1 = realBuffer[i1];
    float im1 = imagBuffer[i1];
    float c = M_SQRT1_2_F;
    float s = M_SQRT1_2_F;

    realBuffer[i1] = (r1 * c) - (im1 * s);
    imagBuffer[i1] = (r1 * s) + (im1 * c);
}

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

// RX: |psi'> = exp(-i * theta/2 * X) |psi|
kernel void rx_gate(device float* realBuffer [[buffer(0)]],
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

    // psi0' = cosH * psi0 - i * sinH * psi1
    realBuffer[i0] = (cosH * r0) + (sinH * im1);
    imagBuffer[i0] = (cosH * im0) - (sinH * r1);

    // psi1' = -i * sinH * psi0 + cosH * psi1
    realBuffer[i1] = (sinH * im0) + (cosH * r1);
    imagBuffer[i1] = (-sinH * r0) + (cosH * im1);
}

// RY: exp(-i * theta/2 * Y)
kernel void ry_gate(device float* realBuffer [[buffer(0)]],
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

    realBuffer[i0] = (cosH * r0) - (sinH * r1);
    imagBuffer[i0] = (cosH * im0) - (sinH * im1);
    realBuffer[i1] = (sinH * r0) + (cosH * r1);
    imagBuffer[i1] = (sinH * im0) + (cosH * im1);
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

// Controlled-RX: applies RX(theta) to the target only when the control bit is 1.
kernel void crx_gate(device float* realBuffer [[buffer(0)]],
                     device float* imagBuffer [[buffer(1)]],
                     constant uint2& qubits [[buffer(2)]],
                     constant float& theta [[buffer(3)]],
                     uint id [[thread_position_in_grid]]) {

    uint controlQubit = qubits.x;
    uint targetQubit = qubits.y;

    uint mask = (1 << targetQubit) - 1;
    uint i0 = ((id >> targetQubit) << (targetQubit + 1)) | (id & mask);
    uint i1 = i0 | (1 << targetQubit);

    if ((i0 & (1 << controlQubit)) == 0) {
        return;
    }

    float r0 = realBuffer[i0];
    float im0 = imagBuffer[i0];
    float r1 = realBuffer[i1];
    float im1 = imagBuffer[i1];

    float halfTheta = theta * 0.5f;
    float cosH = cos(halfTheta);
    float sinH = sin(halfTheta);

    realBuffer[i0] = (cosH * r0) + (sinH * im1);
    imagBuffer[i0] = (cosH * im0) - (sinH * r1);
    realBuffer[i1] = (sinH * im0) + (cosH * r1);
    imagBuffer[i1] = (-sinH * r0) + (cosH * im1);
}

// Controlled-RY: applies RY(theta) to the target only when the control bit is 1.
kernel void cry_gate(device float* realBuffer [[buffer(0)]],
                     device float* imagBuffer [[buffer(1)]],
                     constant uint2& qubits [[buffer(2)]],
                     constant float& theta [[buffer(3)]],
                     uint id [[thread_position_in_grid]]) {

    uint controlQubit = qubits.x;
    uint targetQubit = qubits.y;

    uint mask = (1 << targetQubit) - 1;
    uint i0 = ((id >> targetQubit) << (targetQubit + 1)) | (id & mask);
    uint i1 = i0 | (1 << targetQubit);

    if ((i0 & (1 << controlQubit)) == 0) {
        return;
    }

    float r0 = realBuffer[i0];
    float im0 = imagBuffer[i0];
    float r1 = realBuffer[i1];
    float im1 = imagBuffer[i1];

    float halfTheta = theta * 0.5f;
    float cosH = cos(halfTheta);
    float sinH = sin(halfTheta);

    realBuffer[i0] = (cosH * r0) - (sinH * r1);
    imagBuffer[i0] = (cosH * im0) - (sinH * im1);
    realBuffer[i1] = (sinH * r0) + (cosH * r1);
    imagBuffer[i1] = (sinH * im0) + (cosH * im1);
}

// Controlled-RZ: applies RZ(theta) to the target only when the control bit is 1.
kernel void crz_gate(device float* realBuffer [[buffer(0)]],
                     device float* imagBuffer [[buffer(1)]],
                     constant uint2& qubits [[buffer(2)]],
                     constant float& theta [[buffer(3)]],
                     uint id [[thread_position_in_grid]]) {

    uint controlQubit = qubits.x;
    uint targetQubit = qubits.y;

    uint mask = (1 << targetQubit) - 1;
    uint i0 = ((id >> targetQubit) << (targetQubit + 1)) | (id & mask);
    uint i1 = i0 | (1 << targetQubit);

    if ((i0 & (1 << controlQubit)) == 0) {
        return;
    }

    float r0 = realBuffer[i0];
    float im0 = imagBuffer[i0];
    float r1 = realBuffer[i1];
    float im1 = imagBuffer[i1];

    float halfTheta = theta * 0.5f;
    float cosH = cos(halfTheta);
    float sinH = sin(halfTheta);

    realBuffer[i0] = r0 * cosH + im0 * sinH;
    imagBuffer[i0] = im0 * cosH - r0 * sinH;
    realBuffer[i1] = r1 * cosH - im1 * sinH;
    imagBuffer[i1] = im1 * cosH + r1 * sinH;
}

// Controlled-Phase CP(theta): phase e^{i theta} on |11> (control AND target = 1).
kernel void cphase_gate(device float* realBuffer [[buffer(0)]],
                        device float* imagBuffer [[buffer(1)]],
                        constant uint2& qubits [[buffer(2)]],
                        constant float& theta [[buffer(3)]],
                        uint id [[thread_position_in_grid]]) {

    uint controlQubit = qubits.x;
    uint targetQubit = qubits.y;

    uint mask = (1 << targetQubit) - 1;
    uint i0 = ((id >> targetQubit) << (targetQubit + 1)) | (id & mask);
    uint i1 = i0 | (1 << targetQubit);

    if ((i1 & (1 << controlQubit)) == 0) {
        return;
    }

    float r1 = realBuffer[i1];
    float im1 = imagBuffer[i1];
    float c = cos(theta);
    float s = sin(theta);

    realBuffer[i1] = (r1 * c) - (im1 * s);
    imagBuffer[i1] = (r1 * s) + (im1 * c);
}

// Multi-controlled X: X on target when every control bit (encoded in controlMask) is 1.
kernel void mcx_gate(device float* realBuffer [[buffer(0)]],
                     device float* imagBuffer [[buffer(1)]],
                     constant uint2& packed [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {

    uint controlMask = packed.x;
    uint targetQubit = packed.y;

    uint mask = (1 << targetQubit) - 1;
    uint i0 = ((id >> targetQubit) << (targetQubit + 1)) | (id & mask);
    uint i1 = i0 | (1 << targetQubit);

    if ((i0 & controlMask) == controlMask) {
        float tempR = realBuffer[i0];
        float tempI = imagBuffer[i0];
        realBuffer[i0] = realBuffer[i1];
        imagBuffer[i0] = imagBuffer[i1];
        realBuffer[i1] = tempR;
        imagBuffer[i1] = tempI;
    }
}

// Multi-controlled Z: phase -1 when every involved qubit (controls + target in fullMask) is 1.
kernel void mcz_gate(device float* realBuffer [[buffer(0)]],
                     device float* imagBuffer [[buffer(1)]],
                     constant uint& fullMask [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {

    if ((id & fullMask) == fullMask) {
        realBuffer[id] = -realBuffer[id];
        imagBuffer[id] = -imagBuffer[id];
    }
}

// S-dagger: |1> *= -i  (inverse of S)
kernel void s_dagger_gate(device float* realBuffer [[buffer(0)]],
                          device float* imagBuffer [[buffer(1)]],
                          constant uint& targetQubit [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {

    uint mask = (1 << targetQubit) - 1;
    uint i0 = ((id >> targetQubit) << (targetQubit + 1)) | (id & mask);
    uint i1 = i0 | (1 << targetQubit);

    float r1 = realBuffer[i1];
    float im1 = imagBuffer[i1];

    realBuffer[i1] = im1;
    imagBuffer[i1] = -r1;
}

// T-dagger: |1> *= e^{-i pi/4}  (inverse of T)
kernel void t_dagger_gate(device float* realBuffer [[buffer(0)]],
                          device float* imagBuffer [[buffer(1)]],
                          constant uint& targetQubit [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {

    uint mask = (1 << targetQubit) - 1;
    uint i0 = ((id >> targetQubit) << (targetQubit + 1)) | (id & mask);
    uint i1 = i0 | (1 << targetQubit);

    float r1 = realBuffer[i1];
    float im1 = imagBuffer[i1];
    float c = M_SQRT1_2_F;
    float s = M_SQRT1_2_F;

    realBuffer[i1] = (r1 * c) + (im1 * s);
    imagBuffer[i1] = (im1 * c) - (r1 * s);
}

// SX (sqrt(X)): (1/2) * [[1+i, 1-i], [1-i, 1+i]]; SX*SX = X
kernel void sx_gate(device float* realBuffer [[buffer(0)]],
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

    realBuffer[i0] = 0.5f * (r0 - im0 + r1 + im1);
    imagBuffer[i0] = 0.5f * (r0 + im0 - r1 + im1);
    realBuffer[i1] = 0.5f * (r0 + im0 + r1 - im1);
    imagBuffer[i1] = 0.5f * (-r0 + im0 + r1 + im1);
}

// SX-dagger (sqrt(X)^dagger): (1/2) * [[1-i, 1+i], [1+i, 1-i]]; conjugate-transpose of SX, SX^dagger*SX = I
kernel void sx_dagger_gate(device float* realBuffer [[buffer(0)]],
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

    realBuffer[i0] = 0.5f * (r0 + im0 + r1 - im1);
    imagBuffer[i0] = 0.5f * (-r0 + im0 + r1 + im1);
    realBuffer[i1] = 0.5f * (r0 - im0 + r1 + im1);
    imagBuffer[i1] = 0.5f * (r0 + im0 - r1 + im1);
}

// P(theta): |1> *= e^{i theta}  (general phase; S = P(pi/2), T = P(pi/4), Z = P(pi))
kernel void phase_gate(device float* realBuffer [[buffer(0)]],
                       device float* imagBuffer [[buffer(1)]],
                       constant uint& targetQubit [[buffer(2)]],
                       constant float& theta [[buffer(3)]],
                       uint id [[thread_position_in_grid]]) {

    uint mask = (1 << targetQubit) - 1;
    uint i0 = ((id >> targetQubit) << (targetQubit + 1)) | (id & mask);
    uint i1 = i0 | (1 << targetQubit);

    float r1 = realBuffer[i1];
    float im1 = imagBuffer[i1];
    float c = cos(theta);
    float s = sin(theta);

    realBuffer[i1] = (r1 * c) - (im1 * s);
    imagBuffer[i1] = (r1 * s) + (im1 * c);
}

// Universal single-qubit gate U(theta, phi, lambda) (Qiskit convention):
//   [[ cos(t/2),            -e^{i*lambda} sin(t/2)        ],
//    [ e^{i*phi} sin(t/2),   e^{i(phi+lambda)} cos(t/2)   ]]
// Recovers X = U(pi,0,pi), H = U(pi/2,0,pi), P(λ) = U(0,0,λ), etc.
kernel void u_gate(device float* realBuffer [[buffer(0)]],
                   device float* imagBuffer [[buffer(1)]],
                   constant uint& targetQubit [[buffer(2)]],
                   constant float3& angles [[buffer(3)]],
                   uint id [[thread_position_in_grid]]) {

    uint mask = (1 << targetQubit) - 1;
    uint i0 = ((id >> targetQubit) << (targetQubit + 1)) | (id & mask);
    uint i1 = i0 | (1 << targetQubit);

    float theta = angles.x;
    float phi = angles.y;
    float lambda = angles.z;

    float c = cos(theta * 0.5f);
    float s = sin(theta * 0.5f);

    float cosL = cos(lambda);
    float sinL = sin(lambda);
    float cosP = cos(phi);
    float sinP = sin(phi);
    float cosPL = cos(phi + lambda);
    float sinPL = sin(phi + lambda);

    float r0 = realBuffer[i0];
    float im0 = imagBuffer[i0];
    float r1 = realBuffer[i1];
    float im1 = imagBuffer[i1];

    realBuffer[i0] = (c * r0) - s * ((cosL * r1) - (sinL * im1));
    imagBuffer[i0] = (c * im0) - s * ((cosL * im1) + (sinL * r1));

    realBuffer[i1] = s * ((cosP * r0) - (sinP * im0)) + c * ((cosPL * r1) - (sinPL * im1));
    imagBuffer[i1] = s * ((cosP * im0) + (sinP * r0)) + c * ((cosPL * im1) + (sinPL * r1));
}

// CZ: phase -1 on |11>. Symmetric in control/target.
kernel void cz_gate(device float* realBuffer [[buffer(0)]],
                    device float* imagBuffer [[buffer(1)]],
                    constant uint2& qubits [[buffer(2)]],
                    uint id [[thread_position_in_grid]]) {

    uint controlQubit = qubits.x;
    uint targetQubit = qubits.y;

    uint mask = (1 << targetQubit) - 1;
    uint i0 = ((id >> targetQubit) << (targetQubit + 1)) | (id & mask);
    uint i1 = i0 | (1 << targetQubit);

    if ((i1 & (1 << controlQubit)) != 0) {
        realBuffer[i1] = -realBuffer[i1];
        imagBuffer[i1] = -imagBuffer[i1];
    }
}

// SWAP: exchange amplitudes of |...q1=0,q2=1...> and |...q1=1,q2=0...>.
// Full-state kernel; each unordered pair is handled exactly once.
kernel void swap_gate(device float* realBuffer [[buffer(0)]],
                      device float* imagBuffer [[buffer(1)]],
                      constant uint2& qubits [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {

    uint q1 = qubits.x;
    uint q2 = qubits.y;

    uint b1 = (id >> q1) & 1u;
    uint b2 = (id >> q2) & 1u;

    if (b1 == 0u && b2 == 1u) {
        uint partner = (id | (1u << q1)) & ~(1u << q2);

        float tr = realBuffer[id];
        float ti = imagBuffer[id];
        realBuffer[id] = realBuffer[partner];
        imagBuffer[id] = imagBuffer[partner];
        realBuffer[partner] = tr;
        imagBuffer[partner] = ti;
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

// Reduces the Born-rule probability of the subspace where `targetQubit == |1>`.
// Each threadgroup performs a tree reduction over 256 elements and writes its partial sum into
kernel void masked_population_reduce(device const float* realBuffer [[buffer(0)]],
                                     device const float* imagBuffer [[buffer(1)]],
                                     constant uint& targetQubit [[buffer(2)]],
                                     constant uint& elementCount [[buffer(3)]],
                                     device float* partials [[buffer(4)]],
                                     uint globalID [[thread_position_in_grid]],
                                     uint localID [[thread_index_in_threadgroup]],
                                     uint groupID [[threadgroup_position_in_grid]],
                                     uint groupSize [[threads_per_threadgroup]]) {
    threadgroup float shared[256];

    float value = 0.0f;
    if (globalID < elementCount && (((globalID >> targetQubit) & 1u) != 0u)) {
        const float r = realBuffer[globalID];
        const float i = imagBuffer[globalID];
        value = (r * r) + (i * i);
    }
    shared[localID] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = groupSize >> 1; stride > 0; stride >>= 1) {
        if (localID < stride) {
            shared[localID] += shared[localID + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (localID == 0) {
        partials[groupID] = shared[0];
    }
}

// Compensated (double-single) inclusive prefix sum building the CDF.
//
// The running value is carried as `(hi, lo)` across both the in-block Hillis–Steele scan and the
// block-offset pass, so probabilities far below the float32 ulp of the accumulated total are no
// longer swallowed. `dataHi/dataLo` hold the per-element CDF; `blockHi/blockLo` hold the per-block
// totals (themselves scanned recursively). `readInputLo` is 0 at the leaf level (input probabilities
// have no compensation term yet) and 1 for the recursive block-sum passes.
//
// Phase 0: block-local inclusive scan, writing each block total into `blockHi/blockLo`.
// Phase 2: add the (exclusive) scanned block offset into every element of the block.
kernel void prefix_sum_probabilities(device float* dataHi [[buffer(0)]],
                                     device float* dataLo [[buffer(1)]],
                                     device float* blockHi [[buffer(2)]],
                                     device float* blockLo [[buffer(3)]],
                                     constant uint& elementCount [[buffer(4)]],
                                     constant uint& phase [[buffer(5)]],
                                     constant uint& readInputLo [[buffer(6)]],
                                     uint globalID [[thread_position_in_grid]],
                                     uint localID [[thread_index_in_threadgroup]],
                                     uint groupID [[threadgroup_position_in_grid]],
                                     uint groupSize [[threads_per_threadgroup]]) {
    threadgroup float sharedHi[256];
    threadgroup float sharedLo[256];

    const uint index = groupID * groupSize + localID;

    if (phase == 0) {
        sharedHi[localID] = (index < elementCount) ? dataHi[index] : 0.0f;
        sharedLo[localID] = (index < elementCount && readInputLo != 0u) ? dataLo[index] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = 1; stride < groupSize; stride <<= 1) {
            // Partner partial sum is read into registers; the compensation term never touches VRAM.
            float addHi = 0.0f;
            float addLo = 0.0f;
            if (localID >= stride) {
                addHi = sharedHi[localID - stride];
                addLo = sharedLo[localID - stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (localID >= stride) {
                float2 sum = df_add(float2(sharedHi[localID], sharedLo[localID]),
                                    float2(addHi, addLo));
                sharedHi[localID] = sum.x;
                sharedLo[localID] = sum.y;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (index < elementCount) {
            dataHi[index] = sharedHi[localID];
            dataLo[index] = sharedLo[localID];
        }

        const uint blockEnd = min((groupID + 1) * groupSize, elementCount);
        if (blockEnd > 0 && index == blockEnd - 1) {
            blockHi[groupID] = sharedHi[localID];
            blockLo[groupID] = sharedLo[localID];
        }
        return;
    }

    if (phase == 2) {
        if (index >= elementCount) {
            return;
        }

        float2 offset = (groupID > 0) ? float2(blockHi[groupID - 1], blockLo[groupID - 1])
                                      : float2(0.0f, 0.0f);
        float2 sum = df_add(float2(dataHi[index], dataLo[index]), offset);
        dataHi[index] = sum.x;
        dataLo[index] = sum.y;
    }
}

// Naive single-precision inclusive prefix sum. NOT used by the production collapse path — it is the
// uncompensated Float32 baseline kept solely so benchmarks can measure the cost of the compensated
// scan against the plain one under identical memory access patterns.
kernel void prefix_sum_probabilities_naive(device float* dataBuffer [[buffer(0)]],
                                           device float* blockSums [[buffer(1)]],
                                           constant uint& elementCount [[buffer(2)]],
                                           constant uint& phase [[buffer(3)]],
                                           uint globalID [[thread_position_in_grid]],
                                           uint localID [[thread_index_in_threadgroup]],
                                           uint groupID [[threadgroup_position_in_grid]],
                                           uint groupSize [[threads_per_threadgroup]]) {
    threadgroup float sharedData[256];
    const uint index = groupID * groupSize + localID;

    if (phase == 0) {
        sharedData[localID] = (index < elementCount) ? dataBuffer[index] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = 1; stride < groupSize; stride <<= 1) {
            float accumulated = 0.0f;
            if (localID >= stride) {
                accumulated = sharedData[localID - stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            sharedData[localID] += accumulated;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

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
        if (index >= elementCount) {
            return;
        }
        const float blockOffset = (groupID > 0) ? blockSums[groupID - 1] : 0.0f;
        dataBuffer[index] += blockOffset;
    }
}

// Binary search over the double-single CDF for the first index whose cumulative probability
// exceeds the (also double-single) dice roll. Comparing in compensated precision is what lets the
// search resolve states whose probability is below the float32 ulp of the running total.
kernel void find_collapsed_state(device const float* cdfHi [[buffer(0)]],
                                 device const float* cdfLo [[buffer(1)]],
                                 constant float2& diceRoll [[buffer(2)]],
                                 constant uint& elementCount [[buffer(3)]],
                                 device uint* collapsedIndex [[buffer(4)]],
                                 uint threadID [[thread_position_in_grid]]) {
    if (threadID != 0 || elementCount == 0) {
        return;
    }

    uint low = 0;
    uint high = elementCount - 1;
    uint result = elementCount - 1;

    while (low <= high) {
        const uint mid = (low + high) >> 1;
        if (df_less(diceRoll, float2(cdfHi[mid], cdfLo[mid]))) {
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

kernel void collapse_state_vector(device float* realBuffer [[buffer(0)]],
                                  device float* imagBuffer [[buffer(1)]],
                                  constant uint& collapsedIndex [[buffer(2)]],
                                  uint id [[thread_position_in_grid]]) {
    if (id == collapsedIndex) {
        realBuffer[id] = 1.0f;
        imagBuffer[id] = 0.0f;
    } else {
        realBuffer[id] = 0.0f;
        imagBuffer[id] = 0.0f;
    }
}

kernel void partial_collapse_state_vector(device float* realBuffer [[buffer(0)]],
                                          device float* imagBuffer [[buffer(1)]],
                                          device const uint* measuredQubits [[buffer(2)]],
                                          constant uint& measuredQubitCount [[buffer(3)]],
                                          constant uint& outcome [[buffer(4)]],
                                          uint id [[thread_position_in_grid]]) {
    for (uint position = 0; position < measuredQubitCount; position++) {
        const uint qubit = measuredQubits[position];
        const uint expected = (outcome >> position) & 1u;
        const uint actual = (id >> qubit) & 1u;
        if (actual != expected) {
            realBuffer[id] = 0.0f;
            imagBuffer[id] = 0.0f;
            return;
        }
    }
}


kernel void reset_qubit_state_vector(device float* realBuffer [[buffer(0)]],
                                     device float* imagBuffer [[buffer(1)]],
                                     constant uint& targetQubit [[buffer(2)]],
                                     uint id [[thread_position_in_grid]]) {
    if ((id >> targetQubit) & 1u) {
        realBuffer[id] = 0.0f;
        imagBuffer[id] = 0.0f;
    }
}

// Amplitude-damping quantum jump: applies the Kraus operator K1 = sqrt(gamma)*|0><1| (sigma^-)
kernel void amplitude_damping_jump(device float* realBuffer [[buffer(0)]],
                                   device float* imagBuffer [[buffer(1)]],
                                   constant uint& targetQubit [[buffer(2)]],
                                   uint id [[thread_position_in_grid]]) {
    uint mask = (1 << targetQubit) - 1;
    uint i0 = ((id >> targetQubit) << (targetQubit + 1)) | (id & mask);
    uint i1 = i0 | (1 << targetQubit);

    realBuffer[i0] = realBuffer[i1];
    imagBuffer[i0] = imagBuffer[i1];
    realBuffer[i1] = 0.0f;
    imagBuffer[i1] = 0.0f;
}

// Amplitude-damping no-jump branch: applies the Kraus operator K0 = diag(1, sqrt(1-gamma)).
// |0> amplitudes are preserved; |1> amplitudes are scaled by `factor` = sqrt(1-gamma).
// A global renormalization (by sqrt(1 - gamma*p1)) follows.
kernel void amplitude_damping_no_jump(device float* realBuffer [[buffer(0)]],
                                      device float* imagBuffer [[buffer(1)]],
                                      constant uint& targetQubit [[buffer(2)]],
                                      constant float& factor [[buffer(3)]],
                                      uint id [[thread_position_in_grid]]) {
    uint mask = (1 << targetQubit) - 1;
    uint i0 = ((id >> targetQubit) << (targetQubit + 1)) | (id & mask);
    uint i1 = i0 | (1 << targetQubit);

    realBuffer[i1] *= factor;
    imagBuffer[i1] *= factor;
}

kernel void normalize_state_vector(device float* realBuffer [[buffer(0)]],
                                   device float* imagBuffer [[buffer(1)]],
                                   constant float& invNorm [[buffer(2)]],
                                   uint id [[thread_position_in_grid]]) {
    realBuffer[id] *= invNorm;
    imagBuffer[id] *= invNorm;
}


