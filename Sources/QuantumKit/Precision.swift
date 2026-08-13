//
//  Precision.swift
//  QuantumKit
//
//  Created by Bugra Acemoglu on 13.06.2026.
//

import Foundation

/// Element type for Metal-resident amplitudes and public probability APIs that mirror GPU storage.
///
/// Metal statevector / density-matrix kernels are compiled and buffered for ``Float32`` only.
/// Selecting ``SimulationPrecision/float64`` does **not** widen Metal buffers; use the CPU engines
/// for Double arithmetic (see ``SimulationPrecision``).
public typealias QFloat = Float32

/// Numeric precision policy for simulation backends.
///
/// - ``float32`` (default): Metal path uses ``QFloat`` / Float32 kernels; CPU engines still
///   accumulate in Double internally and expose ``QFloat`` probabilities for API parity.
/// - ``float64``: Prefer host CPU engines with Double state. Metal Float64 kernels are **not**
///   implemented (would require rewriting shader pipelines and buffer layouts); requesting
///   ``float64`` with ``SimulationDevicePreference/metal`` fails, while ``automatic`` falls back to CPU.
public enum SimulationPrecision: String, Sendable, Equatable, Codable {
    case float32
    case float64
}

public enum SimulationPrecisionError: Error, Equatable {
    /// Metal shaders and buffers are Float32-only; Float64 requires the CPU path.
    case metalFloat64Unsupported
}
