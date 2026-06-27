//
//  GateKernels.metal
//  QuantumKit
//
//  Unitary gate compute kernels (single- and multi-qubit).
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
