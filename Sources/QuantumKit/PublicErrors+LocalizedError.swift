import Foundation

// MARK: - H5 actionable public errors
//
// Additive LocalizedError surfaces for library users. Case identity / Equatable are unchanged;
// only user-facing strings are added. Priority: Metal availability, qubit/budget limits,
// unsupported gates, noise/SV mismatch, shots, transpile topology, PEC/ZNE misconfig.

// MARK: QuantumEngineError

extension QuantumEngineError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .deviceNotFound:
            return "No Metal GPU device is available."
        case .libraryNotFound:
            return "Metal shader library could not be loaded."
        case .functionNotFound(let name):
            return "Metal compute function '\(name)' was not found in the shader library."
        case .pipelineStateCreationFailed:
            return "Failed to create a Metal compute pipeline state."
        case .commandBufferCreationFailed:
            return "Failed to create a Metal command buffer."
        case .commandQueueCreationFailed:
            return "Failed to create a Metal command queue."
        case .qubitCountMismatch(let circuit, let state):
            return "Circuit width \(circuit) does not match statevector width \(state)."
        case .bufferAllocationFailed(let requiredBytes):
            return "Metal buffer allocation failed (needed \(requiredBytes) bytes)."
        case .commandBufferExecutionFailed:
            return "Metal command buffer execution failed."
        case .prefixSumBufferLevelMissing(let level):
            return "Prefix-sum scratch buffer is missing for level \(level)."
        case .zeroStateNorm:
            return "Statevector norm is zero; cannot renormalize or sample."
        case .circuitNotUnitaryOnly:
            return "This path requires a unitary-only circuit (no measure/reset/noise side effects)."
        case .localizedNoiseRequiresDensityMatrixBackend:
            return "Localized gate noise cannot run on the statevector backend."
        case .nonProjectiveMeasurementRequiresDensityMatrixBackend:
            return "Non-projective (dephasing-only) measurement requires a density-matrix backend."
        case .unsupportedGateEncoding(let gate):
            return "Unsupported gate encoding for the Metal statevector engine: \(gate)."
        }
    }

    public var failureReason: String? {
        switch self {
        case .deviceNotFound:
            return "MetalRuntime could not resolve an MTLDevice on this host."
        case .localizedNoiseRequiresDensityMatrixBackend:
            return "Statevector / trajectory unraveling only supports global depolarizing (or no noise)."
        case .nonProjectiveMeasurementRequiresDensityMatrixBackend:
            return "Dephasing-only measurement is a mixed-state operation."
        case .unsupportedGateEncoding:
            return "The gate has no Metal kernel encoding on this backend."
        default:
            return nil
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .deviceNotFound:
            return "No Metal GPU is available on this host. Use SimulationDevicePreference.cpu (CPU backends). Device-free Metal inits (e.g. StateVector(qubitCount:)) only omit passing MTLDevice — they still need a GPU."
        case .libraryNotFound, .functionNotFound, .pipelineStateCreationFailed:
            return "Ensure QuantumKit Metal shader resources (.metalsrc) are bundled with the target."
        case .qubitCountMismatch:
            return "Allocate the statevector with the same qubitCount as the circuit."
        case .bufferAllocationFailed:
            return "Reduce qubit count or increase available GPU memory; check maxPeakMemoryBytes if using SimulationPolicy."
        case .localizedNoiseRequiresDensityMatrixBackend:
            return "Enable a density-matrix backend (makeDensityMatrix / preferDensityMatrixWhenNoisy) for localized noise."
        case .nonProjectiveMeasurementRequiresDensityMatrixBackend:
            return "Switch to a density-matrix backend, or use projective MeasurementMode."
        case .unsupportedGateEncoding:
            return "Transpile/decompose the gate to the native basis, or use a CPU/MPS backend that supports it."
        case .circuitNotUnitaryOnly:
            return "Strip measure/reset instructions or use a full execute path that supports mid-circuit ops."
        default:
            return nil
        }
    }
}

// MARK: DensityMatrixEngineError

extension DensityMatrixEngineError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .deviceNotFound:
            return "No Metal GPU device is available."
        case .commandQueueCreationFailed:
            return "Failed to create a Metal command queue for density-matrix simulation."
        case .commandBufferCreationFailed:
            return "Failed to create a Metal command buffer for density-matrix simulation."
        case .commandBufferExecutionFailed:
            return "Metal density-matrix command buffer execution failed."
        case .functionNotFound(let name):
            return "Metal density-matrix function '\(name)' was not found."
        case .libraryNotFound:
            return "Metal density-matrix shader library could not be loaded."
        case .qubitCountMismatch(let circuit, let matrix):
            return "Circuit width \(circuit) does not match density-matrix width \(matrix)."
        case .unsupportedGate(let gate):
            return "Unsupported gate on the density-matrix engine: \(gate)."
        case .nonUnitaryGateUnsupported(let gate):
            return "Non-unitary gate is unsupported on this density-matrix path: \(gate)."
        case .zeroProbabilityMeasurement(let qubit):
            return "Measurement on qubit \(qubit) has zero probability; cannot collapse."
        case .invalidTraceForRenormalization(let trace):
            return "Density-matrix trace \(trace) is invalid for renormalization."
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .deviceNotFound:
            return "No Metal GPU is available on this host. Use SimulationDevicePreference.cpu / CPU density-matrix backends. Device-free DensityMatrix(qubitCount:) still requires a GPU when using Metal DM."
        case .qubitCountMismatch:
            return "Allocate the density matrix with the same qubitCount as the circuit."
        case .unsupportedGate, .nonUnitaryGateUnsupported:
            return "Decompose the gate to the DM-supported basis, or remove unsupported non-unitary ops."
        default:
            return nil
        }
    }
}

// MARK: StateVectorError / DensityMatrixError

extension StateVectorError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .invalidQubitCount(let n):
            return "Statevector qubitCount must be > 0 (got \(n))."
        case .qubitCountExceedsLimit(let max, let requested):
            return "Statevector qubit count \(requested) exceeds limit \(max)."
        case .stateCountOverflow(let qubitCount):
            return "Statevector dimension overflow for qubitCount \(qubitCount)."
        case .bufferAllocationFailed(let requiredBytes):
            return "Failed to allocate statevector Metal buffers (\(requiredBytes) bytes)."
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .qubitCountExceedsLimit, .stateCountOverflow, .bufferAllocationFailed:
            return "Lower qubitCount, raise SimulationPolicy limits, or increase maxPeakMemoryBytes / available memory."
        case .invalidQubitCount:
            return "Pass a positive qubitCount when constructing the statevector."
        }
    }
}

extension DensityMatrixError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .invalidQubitCount(let n):
            return "Density-matrix qubitCount must be > 0 (got \(n))."
        case .qubitCountExceedsLimit(let max, let requested):
            return "Density-matrix qubit count \(requested) exceeds limit \(max)."
        case .matrixElementCountOverflow(let qubitCount):
            return "Density-matrix element count overflow for qubitCount \(qubitCount)."
        case .bufferAllocationFailed(let requiredBytes):
            return "Failed to allocate density-matrix Metal buffers (\(requiredBytes) bytes)."
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .qubitCountExceedsLimit, .matrixElementCountOverflow, .bufferAllocationFailed:
            return "Lower qubitCount, use trajectory/statevector for wider noisy circuits, or raise maxPeakMemoryBytes."
        case .invalidQubitCount:
            return "Pass a positive qubitCount when constructing the density matrix."
        }
    }
}

// MARK: SimulationPolicyError

extension SimulationPolicyError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .qubitCountExceedsAllLimits(let requested, let svMax, let dmMax):
            return "Qubit count \(requested) exceeds all policy limits (statevector ≤ \(svMax), density-matrix ≤ \(dmMax))."
        case .densityMatrixRequiredButTooWide(let requested, let max):
            return "Density-matrix simulation is required for this noise, but width \(requested) exceeds DM limit \(max)."
        case .trajectoryRequiredButTooWide(let requested, let max):
            return "Trajectory simulation is required, but width \(requested) exceeds trajectory/SV limit \(max)."
        case .estimatedMemoryExceedsBudget(let estimated, let budget):
            return "Estimated peak memory \(estimated) bytes exceeds maxPeakMemoryBytes budget \(budget)."
        case .memoryBudgetRequiresQubitCount:
            return "maxPeakMemoryBytes is set but qubitCount was not provided to the factory."
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .qubitCountExceedsAllLimits:
            return "Reduce circuit width or raise statevectorQubitLimit / densityMatrixQubitLimit on SimulationPolicy."
        case .densityMatrixRequiredButTooWide:
            // Thrown when DM does not fit AND (noise is not trajectory-compatible OR
            // preferTrajectoryWhenDensityMatrixTooWide is false). Do not tell callers to
            // "enable trajectory" unconditionally.
            return "Raise densityMatrixQubitLimit (or cpuDensityMatrixQubitLimit on CPU) or reduce circuit width. Trajectory is only an alternative when preferTrajectoryWhenDensityMatrixTooWide is true and NoiseModel.supportsTrajectorySimulation; otherwise simplify noise so density-matrix is not required — trajectory will not help for incompatible noise."
        case .trajectoryRequiredButTooWide:
            return "Reduce width or raise statevector / CPU statevector limits on SimulationPolicy."
        case .estimatedMemoryExceedsBudget:
            return "Increase maxPeakMemoryBytes, reduce qubitCount, or choose a lighter simulation method."
        case .memoryBudgetRequiresQubitCount:
            return "Pass qubitCount: to makeStatevector / makeDensityMatrix / makeRecommended when a memory budget is set."
        }
    }
}

extension SimulationPrecisionError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .metalFloat64Unsupported:
            return "Metal backends support Float32 only; Float64 is unavailable on the GPU path."
        }
    }

    public var recoverySuggestion: String? {
        "Use SimulationDevicePreference.cpu (or .automatic) with SimulationPrecision.float64."
    }
}

// MARK: CPU / trajectory

extension CPUEngineError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .qubitCountMismatch(let circuit, let state):
            return "Circuit width \(circuit) does not match CPU state width \(state)."
        case .qubitCountExceedsLimit(let max, let requested):
            return "CPU qubit count \(requested) exceeds limit \(max)."
        case .invalidQubitCount(let n):
            return "CPU qubitCount must be > 0 (got \(n))."
        case .zeroStateNorm:
            return "CPU statevector norm is zero."
        case .zeroProbabilityMeasurement(let qubit):
            return "Measurement on qubit \(qubit) has zero probability."
        case .unsupportedOnCPU(let reason):
            return "Unsupported on CPU backend: \(reason)."
        case .localizedNoiseRequiresDensityMatrixBackend:
            return "Localized gate noise cannot run on the CPU statevector backend."
        case .nonProjectiveMeasurementRequiresDensityMatrixBackend:
            return "Non-projective measurement requires a density-matrix backend."
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .qubitCountExceedsLimit:
            return "Reduce qubitCount or raise cpuStatevectorQubitLimit / cpuDensityMatrixQubitLimit."
        case .localizedNoiseRequiresDensityMatrixBackend:
            return "Enable a density-matrix backend for localized noise."
        case .nonProjectiveMeasurementRequiresDensityMatrixBackend:
            return "Switch to a density-matrix backend, or use projective MeasurementMode."
        case .unsupportedOnCPU:
            return "Use a Metal backend if available, or decompose to CPU-supported operations."
        default:
            return nil
        }
    }
}

extension TrajectoryBackendError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .shotsRequired:
            return "TrajectoryBackend.run requires a positive QuantumRunOptions.shots count."
        case .unsupportedUnderlyingBackend:
            return "TrajectoryBackend can only wrap Metal or CPU statevector backends."
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .shotsRequired:
            return "Set QuantumRunOptions.shots > 0, or call averageProbabilities(circuit:trajectories:seed:noise:) for an explicit ensemble without a histogram."
        case .unsupportedUnderlyingBackend:
            return "Wrap StatevectorBackend or CPUStatevectorBackend (not DM/MPS/stabilizer)."
        }
    }
}

// MARK: Estimator / Sampler / Measurement

extension EstimatorError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .unsupportedBackend:
            return "Estimator does not support this backend type."
        case .invalidShotCount(let shots):
            return "Invalid Estimator shot count \(shots); shots must be > 0 when sampling."
        case .invalidPrecision(let precision):
            return "Invalid Estimator precision \(precision); precision must be > 0."
        case .shotsRequiredForTrajectory:
            return "Trajectory expectations need an explicit shot / ensemble budget."
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .unsupportedBackend:
            return "Use statevector, density-matrix, trajectory, or MPS backends supported by Estimator."
        case .invalidShotCount:
            return "Set EstimatorOptions.shots (or QuantumRunOptions.shots) to a positive integer."
        case .invalidPrecision:
            return "Pass a positive precision, or set shots explicitly instead."
        case .shotsRequiredForTrajectory:
            return "Set EstimatorOptions.shots or QuantumRunOptions.shots for trajectory backends."
        }
    }
}

extension SamplerError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .unsupportedBackend:
            return "Sampler does not support this backend type."
        }
    }

    public var recoverySuggestion: String? {
        "Use statevector, density-matrix, trajectory, or host MPS backends."
    }
}

extension QuantumMeasurementError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .invalidShotCount(let shots):
            return "Invalid shot count \(shots); shots must be > 0."
        case .emptyQubitSelection:
            return "Measurement qubit selection is empty."
        case .qubitIndexOutOfBounds(let index, let qubitCount):
            return "Qubit index \(index) is out of bounds for width \(qubitCount)."
        case .invalidPauliString(let string):
            return "Invalid Pauli string '\(string)'."
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .invalidShotCount:
            return "Pass a positive shots value in QuantumRunOptions or the measurement API."
        case .emptyQubitSelection:
            return "Provide at least one qubit index to measure."
        case .qubitIndexOutOfBounds:
            return "Use qubit indices in 0..<qubitCount."
        case .invalidPauliString:
            return "Use characters from {I,X,Y,Z} with length matching the register width."
        }
    }
}

// MARK: Transpiler

extension TranspilerError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .unsupportedGate(let gate):
            return "Transpiler cannot decompose unsupported gate: \(gate)."
        case .unsupportedGateForBasis(let gate, let basis):
            return "Gate \(gate) is not supported for basis \(basis)."
        case .mismatchedQubitCounts(let original, let transpiled):
            return "Transpiled circuit width \(transpiled) does not match original width \(original)."
        case .invalidCouplingMap(let reason):
            return "Invalid coupling map: \(reason)."
        case .invalidLayout(let reason):
            return "Invalid layout: \(reason)."
        case .circuitWiderThanDevice(let circuitQubits, let deviceQubits):
            return "Circuit width \(circuitQubits) exceeds device width \(deviceQubits)."
        case .qubitsNotConnected(let a, let b):
            return "Qubits \(a) and \(b) are not connected in the coupling map."
        case .routingRequiresTwoQubitGates(let gate):
            return "Routing requires two-qubit gates; cannot route with \(gate)."
        case .unboundParameters(let names):
            return "Circuit has unbound parameters: \(names.joined(separator: ", "))."
        case .circuitExceedsMaxDepth(let depth, let maxDepth):
            return "Circuit depth \(depth) exceeds maxDepth \(maxDepth)."
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .unsupportedGate, .unsupportedGateForBasis:
            return "Choose a broader BasisGateSet or pre-decompose the gate."
        case .invalidCouplingMap:
            return "Provide a connected CouplingMap with valid undirected edges for the device topology."
        case .invalidLayout:
            return "Provide a bijective initial layout covering all circuit qubits onto device qubits."
        case .circuitWiderThanDevice:
            return "Use a wider device coupling map, or shrink the logical circuit."
        case .qubitsNotConnected:
            return "Add SWAP routing (device-aware transpile) or fix the coupling map connectivity."
        case .unboundParameters:
            return "Bind all QuantumParameter values before transpiling or executing."
        case .circuitExceedsMaxDepth:
            return "Raise maxDepth in transpile options, or simplify the circuit."
        default:
            return nil
        }
    }
}

extension CompilerPassRegistryError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .emptyID:
            return "CompilerPassFactory id must be a non-empty string."
        case .duplicateID(let id):
            return "CompilerPassFactory id \"\(id)\" is already registered."
        case .unknownID(let id):
            return "No CompilerPassFactory registered for id \"\(id)\"."
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .emptyID:
            return "Provide a non-empty stable id (prefer reverse-DNS, e.g. com.example.fuse)."
        case .duplicateID:
            return "Unregister the existing id, use registerReplacing, or choose a different id."
        case .unknownID:
            return "Register a CompilerPassFactory for that id before makePass / makePassManager."
        }
    }
}

// MARK: PEC / ZNE / Pauli twirling

extension PECError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .incompatibleWithZNE:
            return "PEC and ZNE cannot be enabled together in ResilienceOptions."
        case .missingNoiseModel:
            return "PEC requires a NoiseModel on the run options."
        case .unsupportedNoiseModel(let detail):
            return "PEC unsupported noise model: \(detail)."
        case .nonInvertibleChannel(let detail):
            return "PEC channel is not invertible: \(detail)."
        case .unsupportedMultiQubitGate(let detail):
            return "PEC does not support this multi-qubit gate site: \(detail)."
        case .emptyCircuitNoPECSites:
            return "Circuit has no PEC quasi-probability sites."
        case .invalidCircuitSampleCount(let count):
            return "Invalid PEC circuitSamples \(count); must be > 0."
        case .circuitSamplesExceedShots(let samples, let shots):
            return "PEC circuitSamples \(samples) exceeds Estimator shot budget \(shots)."
        case .pauliChannelMismatch(let expectedP, let actualDepolarizing):
            return "PEC Pauli-channel mismatch (expected p≈\(expectedP), depolarizing \(actualDepolarizing))."
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .incompatibleWithZNE:
            return "Disable ZNE or PEC in ResilienceOptions; enable only one mitigation method."
        case .missingNoiseModel:
            return "Attach a NoiseModel with supported local depolarizing / Pauli channels before enabling PEC."
        case .unsupportedNoiseModel, .nonInvertibleChannel:
            return "Use invertible local depolarizing (p < 3/4) or matching Pauli-channel rates supported by PEC."
        case .unsupportedMultiQubitGate:
            return "Decompose multi-qubit gates so PEC sites are 1Q (or supported) channels only."
        case .invalidCircuitSampleCount:
            return "Set PECOptions.circuitSamples to a positive integer."
        case .circuitSamplesExceedShots:
            return "Lower circuitSamples or increase EstimatorOptions.shots so samples ≤ shots."
        case .emptyCircuitNoPECSites:
            return "Ensure the circuit contains noisy gate sites covered by the NoiseModel."
        case .pauliChannelMismatch:
            return "Align NoiseModel Pauli rates with the depolarizing probability used for PEC."
        }
    }
}

extension PauliTwirlingError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .incompatibleWithZNE:
            return "Pauli twirling and ZNE cannot be enabled together in ResilienceOptions."
        case .incompatibleWithPEC:
            return "Pauli twirling and PEC cannot be enabled together in ResilienceOptions."
        case .emptyCircuitNoTwirlSites:
            return "Circuit has no Clifford 1Q/2Q sites for Pauli twirling."
        case .invalidEnsembleSize(let count):
            return "Invalid Pauli twirling ensembleSize \(count); must be > 0."
        case .ensembleExceedsShots(let ensemble, let shots):
            return "Pauli twirling ensembleSize \(ensemble) exceeds Estimator shot budget \(shots)."
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .incompatibleWithZNE, .incompatibleWithPEC:
            return "Enable only one of active ZNE, PEC, or Pauli twirling in ResilienceOptions."
        case .emptyCircuitNoTwirlSites:
            return "Include at least one Clifford 1Q/2Q gate (H, Pauli, S/SX, CX/CZ/SWAP/…)."
        case .invalidEnsembleSize:
            return "Set PauliTwirlingOptions.ensembleSize to a positive integer."
        case .ensembleExceedsShots:
            return "Lower ensembleSize or increase EstimatorOptions.shots so ensemble ≤ shots."
        }
    }
}

extension ZNEError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .insufficientScaleFactors(let count):
            return "ZNE needs at least 2 scale factors (got \(count))."
        case .nonDistinctScaleFactors:
            return "ZNE scale factors must be pairwise distinct."
        case .nonFiniteScaleFactor:
            return "ZNE scale factors must be finite."
        case .negativeScaleFactor:
            return "ZNE scale factors must be ≥ 0."
        case .scaleValueCountMismatch(let scales, let values):
            return "ZNE scale factor count \(scales) does not match value count \(values)."
        case .singularLinearFit:
            return "ZNE linear fit is singular (degenerate scale factors)."
        case .missingGlobalDepolarizing:
            return "Global-depolarizing ZNE requires NoiseModel.depolarizingProbability > 0."
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .insufficientScaleFactors, .nonDistinctScaleFactors, .singularLinearFit:
            return "Provide at least two distinct finite scale factors (e.g. 1.0 and 2.0) in ZNEOptions."
        case .nonFiniteScaleFactor, .negativeScaleFactor:
            return "Use finite scale factors ≥ 0."
        case .scaleValueCountMismatch:
            return "Pass one expectation value per scale factor."
        case .missingGlobalDepolarizing:
            return "Set NoiseModel.depolarizingProbability > 0, or use a noise model compatible with ZNE scaling."
        }
    }
}

// MARK: MPS / Stabilizer (extend existing CustomStringConvertible)

extension MPSError: LocalizedError {
    public var errorDescription: String? { description }

    public var recoverySuggestion: String? {
        switch self {
        case .qubitCountExceedsLimit, .amplitudeExportTooWide:
            return "Reduce qubitCount or raise MPSConfiguration limits / maxAmplitudeExportQubits."
        case .unsupportedGate, .unsupportedMultiQubitGate:
            return "Decompose to 1–2 qubit gates supported by the MPS backend."
        case .noiseNotSupported:
            return "Run noiseless circuits on MPS, or use density-matrix / trajectory for noise."
        case .invalidBondDimension:
            return "Set MPSConfiguration.maxBondDimension to an integer ≥ 1."
        case .invalidQubitCount:
            return "Pass a positive qubitCount."
        case .qubitCountMismatch:
            return "Allocate or reuse an MPS state whose qubitCount matches the circuit width."
        case .svdFailed:
            // LAPACK zheev_ info ≠ 0 (numerical SVD failure), not χ truncation.
            return "Retry the run and check that the MPS state remains finite; for small qubitCount fall back to a CPU statevector or density-matrix backend. Changing maxBondDimension does not fix this LAPACK failure."
        }
    }
}

extension StabilizerError: LocalizedError {
    public var errorDescription: String? { description }

    public var recoverySuggestion: String? {
        switch self {
        case .nonCliffordGate:
            return "Use Clifford+measure gates only, or switch to statevector/DM/MPS for non-Clifford circuits."
        case .noiseNotSupported:
            return "Remove NoiseModel for stabilizer simulation, or use a noisy DM/trajectory backend."
        case .nonProjectiveMeasurementNotSupported:
            return "Use projective measurement mode with the stabilizer backend."
        case .qubitCountExceedsLimit:
            return "Reduce qubitCount or raise the stabilizer width limit."
        case .invalidQubitCount:
            return "Pass a positive qubitCount."
        case .qubitCountMismatch:
            return "Allocate or reuse a stabilizer tableau whose qubitCount matches the circuit width."
        }
    }
}

// MARK: - H5 complete residuals (circuit / IR / parameters / batch)

extension QuantumCircuitError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .invalidQubitCount(let n):
            return "Circuit qubitCount must be > 0 (got \(n))."
        case .qubitIndexOutOfBounds(let index, let qubitCount):
            return "Qubit index \(index) is out of bounds for circuit width \(qubitCount)."
        case .invalidAlgorithmParameter(let reason):
            return "Invalid circuit / gate parameter: \(reason)."
        case .circuitNotUnitary:
            return "Operation requires a unitary-only circuit (no measure/reset/initialize/c_if/while_c)."
        case .invalidComposition(let reason):
            return "Invalid circuit composition: \(reason)."
        case .unsupportedControlledGate(let reason):
            return "Cannot lift gate to a controlled form: \(reason)."
        case .maxLoopIterationsExceeded(let maxIterations):
            return "while_c exceeded maxIterations (\(maxIterations)) without satisfying the exit condition."
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .invalidQubitCount:
            return "Construct QuantumCircuit with a positive qubitCount."
        case .qubitIndexOutOfBounds:
            return "Use qubit indices in 0..<qubitCount."
        case .invalidAlgorithmParameter:
            return "Fix control/target overlap or empty control lists before appending the gate."
        case .circuitNotUnitary:
            return "Strip measure/reset/initialize/c_if/while_c, or use an API that accepts mid-circuit ops."
        case .invalidComposition:
            return "Align qubit maps / classical-register widths when using append, compose, or tensor."
        case .unsupportedControlledGate:
            return "Use a gate with a native controlled encoding, or decompose before control-lifting."
        case .maxLoopIterationsExceeded:
            return "Raise maxIterations, or ensure the loop body updates the classical register so the while condition becomes false."
        }
    }
}

extension CircuitIRError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .unsupportedSchemaVersion(let found, let supported):
            return "Unsupported CircuitIR schemaVersion \(found); this library supports \(supported)."
        case .metadataLengthMismatch(let metadataCount, let gateCount):
            return "instructionMetadata length \(metadataCount) does not match gates length \(gateCount)."
        case .invalidCircuit(let reason):
            return "Invalid circuit IR payload: \(reason)."
        case .controlFlowNotSerialized(let op):
            return "Control-flow op '\(op)' cannot be encoded under CircuitIR schema v1 (G10 lite; in-memory only)."
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .unsupportedSchemaVersion:
            return "Re-export with CircuitIRSchema.current schemaVersion, or upgrade QuantumKit."
        case .metadataLengthMismatch:
            return "Provide instructionMetadata with one optional entry per gate (same length as gates)."
        case .invalidCircuit:
            return "Fix qubit bounds / classical-register declarations before encoding or decoding IR."
        case .controlFlowNotSerialized:
            return "Keep while_c circuits in memory, or expand/unroll before encode until a future schema bump."
        }
    }
}

extension ParameterBindingError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .missingBinding(let name):
            return "Missing parameter binding for '\(name)'."
        case .unboundParameters(let names):
            return "Unbound parameters: \(names.sorted().joined(separator: ", "))."
        case .circuitContainsUnboundParameters(let names):
            return "Circuit still has unbound parameters: \(names.sorted().joined(separator: ", "))."
        }
    }

    public var recoverySuggestion: String? {
        "Bind all QuantumParameter values (e.g. bind / assign) before evaluating expressions or executing."
    }
}

extension BatchExecutionError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .invalidBatchSize(let capacity):
            return "Batch capacity must be > 0 (got \(capacity))."
        }
    }

    public var recoverySuggestion: String? {
        "Pass a positive capacity to StateVectorBatch(qubitCount:capacity:)."
    }
}

extension StateVectorInitializationError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .invalidBasisIndex(let index, let stateCount):
            return "Computational-basis index \(index) is outside 0..<\(stateCount)."
        case .amplitudeCountMismatch(let expected, let actual):
            return "Amplitude count \(actual) does not match statevector dimension \(expected)."
        case .nonUnitNorm(let squaredNorm):
            return "Statevector amplitudes are not unit norm (‖ψ‖² = \(squaredNorm))."
        case .invalidQubitSubset(let reason):
            return "Invalid qubit subset for statevector initialization: \(reason)."
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .invalidBasisIndex:
            return "Use initializeComputationalBasis(index:) with index in 0..<stateCount."
        case .amplitudeCountMismatch:
            return "Pass exactly 2^qubitCount ComplexAmplitude values to initialize(amplitudes:)."
        case .nonUnitNorm:
            return "Normalize amplitudes so Σ|a|² ≈ 1 before initialize(amplitudes:)."
        case .invalidQubitSubset:
            return "Provide a valid qubit subset matching the partial-register initialize API."
        }
    }
}

extension DensityMatrixInitializationError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .invalidQubitSubset(let reason):
            return "Invalid qubit subset for density-matrix initialization: \(reason)."
        case .amplitudeCountMismatch(let expected, let actual):
            return "Amplitude count \(actual) does not match density-matrix dimension \(expected)."
        case .nonUnitNorm(let squaredNorm):
            return "Pure-state amplitudes for ρ are not unit norm (‖ψ‖² = \(squaredNorm))."
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .amplitudeCountMismatch:
            return "Pass exactly 2^qubitCount ComplexAmplitude values to DensityMatrix.initialize(amplitudes:)."
        case .nonUnitNorm:
            return "Normalize amplitudes so Σ|a|² ≈ 1 before building |ψ⟩⟨ψ|."
        case .invalidQubitSubset:
            return "Provide a valid qubit subset matching the partial-register initialize API."
        }
    }
}

// MARK: Noise / readout / post-selection

extension QuantumChannelError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .emptyKrausSet:
            return "Kraus operator set is empty."
        case .invalidKrausDimension(let operatorIndex, let count):
            return "Kraus operator \(operatorIndex) has invalid length \(count) (expected 4 for 1-qubit)."
        case .pauliProbabilitiesExceedOne(let sum):
            return "Pauli-channel probabilities sum to \(sum), which exceeds 1."
        case .invalidProcessMatrixDimension(let count, let expected):
            return "Process matrix length \(count) is invalid (expected \(expected))."
        case .multiQubitProcessMatrixUnsupported(let qubitCount):
            return "Multi-qubit process-matrix import is unsupported (qubitCount \(qubitCount))."
        case .choiNotHermitian(let maxDeviation):
            return "Choi matrix failed hermiticity check (max |H−H†| = \(maxDeviation))."
        case .choiNotPositiveSemidefinite(let minimumEigenvalue):
            return "Choi matrix is not positive semidefinite (min eigenvalue \(minimumEigenvalue))."
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .emptyKrausSet, .invalidKrausDimension:
            return "Supply a non-empty 1-qubit Kraus set via QuantumChannel.fromKraus1Q (each operator: length-4 row-major 2×2)."
        case .pauliProbabilitiesExceedOne:
            return "Choose px, py, pz ≥ 0 with px+py+pz ≤ 1 via QuantumChannel.makePauliChannel."
        case .invalidProcessMatrixDimension:
            return "Pass a length-16 row-major 4×4 process / Choi matrix for 1-qubit import."
        case .multiQubitProcessMatrixUnsupported:
            return "Import only 1-qubit process matrices, or build multi-qubit noise via other QuantumChannel cases."
        case .choiNotHermitian, .choiNotPositiveSemidefinite:
            return "Correct the Choi / process matrix so it is Hermitian and positive semidefinite within tolerance."
        }
    }
}

extension ReadoutConfusionError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .invalidDimension(let expected, let actual):
            return "Readout confusion matrix dimension mismatch (expected \(expected), got \(actual))."
        case .emptyMatrix:
            return "ReadoutConfusionMatrix requires qubitCount > 0."
        case .nonStochasticRow(let row):
            return "Readout confusion row \(row) is not stochastic (row probabilities must sum to ~1)."
        case .probabilityOutOfRange:
            return "Readout confusion entry is outside [0, 1]."
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .emptyMatrix:
            return "Construct ReadoutConfusionMatrix with a positive qubitCount."
        case .invalidDimension:
            return "Provide a 2ⁿ×2ⁿ row-stochastic matrix matching qubitCount."
        case .nonStochasticRow, .probabilityOutOfRange:
            return "Ensure each row is a probability distribution with entries in [0, 1]."
        }
    }
}

extension ReadoutMitigationError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .qubitCountMismatch(let histogram, let matrix):
            return "Readout mitigation width mismatch (histogram \(histogram), matrix \(matrix))."
        case .singularConfusionMatrix:
            return "Readout confusion matrix is singular; inverse mitigation failed."
        case .nonPositivePreparedMass:
            return "Inverse readout mitigation produced no positive prepared mass after clamping."
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .qubitCountMismatch:
            return "Pass the same qubitCount used by ShotCounts and ReadoutConfusionMatrix to ReadoutMitigation.apply."
        case .singularConfusionMatrix:
            return "Use a full-rank ReadoutConfusionMatrix (or independent bit-flip rates) before mitigation."
        case .nonPositivePreparedMass:
            return "Check the confusion matrix / histogram consistency; near-singular C can yield empty prepared mass."
        }
    }
}

extension PostSelectionError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .invalidBitValue(let value):
            return "Post-selection bit value must be 0 or 1 (got \(value))."
        case .qubitOutOfRange(let qubit, let qubitCount):
            return "Post-selection qubit \(qubit) is out of range for width \(qubitCount)."
        case .emptyKeepSet(let discardedShots):
            return "Post-selection kept zero shots (discarded \(discardedShots))."
        case .missingShotCounts:
            return "Post-selection requires SamplerResult.shotCounts (exact Born path has none)."
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .invalidBitValue:
            return "Use bit values 0 or 1 in the post-selection predicate."
        case .qubitOutOfRange:
            return "Select qubits in 0..<qubitCount."
        case .emptyKeepSet:
            return "Relax the predicate, increase shots, or use EmptyKeepSetPolicy.emptyResult."
        case .missingShotCounts:
            return "Run Sampler with a positive shot budget so SamplerResult.shotCounts is present."
        }
    }
}

// MARK: Gradients / checkpoints / DAG / ancilla / ordering

extension GradientCalculatorError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .unsupportedBackend:
            return "GradientCalculator does not support this backend type."
        case .noDifferentiableParameters:
            return "Circuit has no differentiable parameters for gradient evaluation."
        case .missingParameterBinding(let name):
            return "Missing parameter binding for gradient parameter '\(name)'."
        case .adjointRequiresUnitaryNoiseFreeCircuit:
            return "Adjoint differentiation requires a unitary, noise-free circuit."
        case .adjointUnsupportedGate(let gate):
            return "Adjoint differentiator does not support gate: \(gate)."
        case .adjointRequiresStatevectorBackend:
            return "Adjoint differentiation requires a statevector backend."
        case .hessianRequiresParameterShift:
            return "Hessian / second-order differentiation requires DifferentiationMethod.parameterShift."
        case .heterogeneousParameterScale(let parameter, let scales):
            return "Parameter '\(parameter)' appears with unequal linear scales \(scales)."
        case .unsupportedParameterShiftGate(let parameter):
            return "Parameter '\(parameter)' is not a linear angle in a supported rotation gate for parameter-shift."
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .unsupportedBackend:
            return "Use a statevector (or other GradientCalculator-supported) backend."
        case .noDifferentiableParameters:
            return "Attach QuantumParameter expressions to rotation angles before differentiating."
        case .missingParameterBinding:
            return "Bind every named QuantumParameter before calling GradientCalculator / ObservableDifferentiator."
        case .adjointRequiresUnitaryNoiseFreeCircuit:
            return "Remove measure/reset/noise, or switch DifferentiationMethod to .parameterShift."
        case .adjointUnsupportedGate:
            return "Decompose unsupported gates, or use DifferentiationMethod.parameterShift."
        case .adjointRequiresStatevectorBackend:
            return "Wrap a StatevectorBackend / CPUStatevectorBackend, or use parameter-shift."
        case .hessianRequiresParameterShift:
            return "Set DifferentiationMethod.parameterShift for Hessian evaluation."
        case .heterogeneousParameterScale:
            return "Use a single homogeneous scale for each parameter across all gates (see ParameterShift docs)."
        case .unsupportedParameterShiftGate:
            return "Express the parameter as a linear angle on RX/RY/RZ/RXX/RYY/RZZ (or supported scaled forms)."
        }
    }
}

extension CheckpointError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .qubitCountMismatch(let expected, let actual):
            return "Checkpoint qubitCount mismatch (expected \(expected), got \(actual))."
        case .elementCountMismatch(let expected, let actual):
            return "Checkpoint amplitude/element count mismatch (expected \(expected), got \(actual))."
        case .instructionIndexOutOfBounds(let index, let gateCount):
            return "CircuitRunState fromInstruction \(index) is out of bounds for gateCount \(gateCount)."
        case .instructionRangeInvalid(let from, let to, let gateCount):
            return "Invalid CircuitRunState instruction range [\(from), \(to)) for gateCount \(gateCount)."
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .qubitCountMismatch, .elementCountMismatch:
            return "Restore snapshots onto a state with matching qubitCount / element layout."
        case .instructionIndexOutOfBounds, .instructionRangeInvalid:
            return "Set CircuitRunState.fromInstruction / toInstruction within 0...gateCount (to exclusive)."
        }
    }
}

extension CircuitExecutionCancellationError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .cancelled:
            return "Circuit execution was cancelled."
        }
    }

    public var recoverySuggestion: String? {
        "Discard the in-flight quantum state and allocate a fresh StateVector / DensityMatrix / CPU state before retrying."
    }
}

extension DAGCircuitError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .cycleDetected:
            return "DAGCircuit has a dependency cycle."
        case .unknownNode(let id):
            return "Unknown DAG node id \(id.rawValue)."
        case .invalidCircuit(let reason):
            return "Invalid DAGCircuit: \(reason)."
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .cycleDetected:
            return "Remove cyclic qubit/classical edges before topological sort / flatten."
        case .unknownNode:
            return "Use node ids returned by DAGCircuit insertion APIs."
        case .invalidCircuit:
            return "Pass a positive qubitCount and valid gate dependencies when building the DAG."
        }
    }
}

extension AncillaAllocationError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .ancillaAllocationDisabled(let strategy):
            return "Ancilla allocation is disabled for controlled-gate synthesis strategy \(strategy)."
        case .insufficientAncillas(let required, let available):
            return "Need \(required) ancillas but only \(available) are available."
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .ancillaAllocationDisabled:
            return "Enable ancilla allocation in TranspileOptions / ControlledGateSynthesisStrategy, or choose a no-ancilla strategy."
        case .insufficientAncillas:
            return "Widen the circuit / device layout, or lower multi-controlled gate fan-in so fewer ancillas are required."
        }
    }
}

extension QubitBitOrderingError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .invalidBitstringLength(let expected, let actual):
            return "Bitstring length \(actual) does not match qubitCount \(expected)."
        case .invalidBitCharacter(let ch):
            return "Invalid bitstring character '\(ch)'; expected '0' or '1'."
        case .indexOutOfRange(let index, let qubitCount):
            return "Basis index \(index) is out of range for qubitCount \(qubitCount)."
        case .invalidBitValue(let value):
            return "Invalid bit value \(value); expected 0 or 1."
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .invalidBitstringLength:
            return "Pass a bitstring whose length equals qubitCount to QubitBitOrdering.index(fromBitstring:)."
        case .invalidBitCharacter, .invalidBitValue:
            return "Use only characters / values 0 and 1."
        case .indexOutOfRange:
            return "Use indices in 0..<2^qubitCount with QubitBitOrdering.bitstring(forIndex:)."
        }
    }
}

extension OperatorError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .invalidPauliLabel(let label):
            return "Invalid sparse Pauli label '\(label)'."
        case .duplicateQubitInTerm(let qubit, let label):
            return "Duplicate qubit \(qubit) in Pauli term label '\(label)'."
        }
    }

    public var recoverySuggestion: String? {
        "Use sparse labels like \"Z0 Z1\" / \"X0\" with distinct qubit indices (PauliTerm.init(coefficient:label:))."
    }
}

extension ClassicalMemoryError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .invalidBitCount(let n):
            return "Classical register bitCount must be > 0 (got \(n))."
        case .registerIndexOutOfBounds(let index, let count):
            return "Classical register index \(index) is out of bounds (count \(count))."
        case .bitOffsetOutOfBounds(let register, let offset, let width):
            return "Bit offset \(offset) is out of bounds for register \(register) (width \(width))."
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .invalidBitCount:
            return "Construct ClassicalRegisterSpec with a positive bitCount."
        case .registerIndexOutOfBounds:
            return "Use register indices in range for ClassicalMemory, or match snapshot widths on restore(from:)."
        case .bitOffsetOutOfBounds:
            return "Use bit offsets in 0..<registerWidths[register]."
        }
    }
}

// MARK: Algorithms / benchmarks / Shor

extension TrotterError: LocalizedError {
    public var errorDescription: String? { description }

    public var recoverySuggestion: String? {
        switch self {
        case .invalidStepCount:
            return "Pass steps ≥ 1 to TrotterEvolution."
        case .invalidQubitCount:
            return "Pass qubitCount ≥ 1 when building Trotter circuits."
        case .qubitIndexOutOfRange:
            return "Ensure every PauliTerm qubit index lies in 0..<qubitCount."
        }
    }
}

extension VariationalAlgorithmError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .invalidLayerCount(let layers):
            return "Variational layer count must be > 0 (got \(layers))."
        case .emptyProblem:
            return "Variational problem is empty (need qubitCount > 0 and a valid IsingGraph / Hamiltonian)."
        case .qubitOutOfRange(let qubit, let qubitCount):
            return "Qubit \(qubit) is out of range for width \(qubitCount)."
        case .duplicateEdge(let a, let b):
            return "Duplicate IsingGraph edge (\(a), \(b))."
        case .missingParameterBinding(let name):
            return "Missing variational parameter binding for '\(name)'."
        case .parameterCountMismatch(let expected, let actual):
            return "Variational parameter count mismatch (expected \(expected), got \(actual))."
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .invalidLayerCount:
            return "Pass a positive layer count to QAOA (or the variational builder)."
        case .emptyProblem:
            return "Construct IsingGraph with qubitCount > 0 and at least a valid edge/field set."
        case .qubitOutOfRange:
            return "Keep edge / field qubit indices in 0..<qubitCount."
        case .duplicateEdge:
            return "Deduplicate undirected edges before constructing IsingGraph."
        case .missingParameterBinding:
            return "Bind all ansatz parameters before VQE / QAOA evaluation."
        case .parameterCountMismatch:
            return "Supply exactly the number of parameters expected by the ansatz / QAOA layers."
        }
    }
}

extension QuantumVolumeError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .invalidQubitCount(let n):
            return "Quantum Volume qubitCount is invalid (got \(n))."
        case .invalidDepth(let depth):
            return "Quantum Volume depth is invalid (got \(depth))."
        case .emptyProbabilities:
            return "Heavy-output / QV probability vector is empty."
        case .probabilityCountMismatch(let expected, let actual):
            return "QV probability count \(actual) does not match expected dimension \(expected)."
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .invalidQubitCount, .invalidDepth:
            return "Pass positive qubitCount and depth to QuantumVolume model-circuit APIs."
        case .emptyProbabilities, .probabilityCountMismatch:
            return "Provide a full 2ⁿ Born probability vector from a noiseless statevector run."
        }
    }
}

extension RandomizedBenchmarkingError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .invalidQubitCount(let n):
            return "Randomized benchmarking qubitCount is invalid (got \(n))."
        case .invalidSequenceLength(let length):
            return "Randomized benchmarking sequenceLength is invalid (got \(length))."
        case .qubitOutOfRange(let qubit, let qubitCount):
            return "RB target qubit \(qubit) is out of range for width \(qubitCount)."
        case .emptyCliffordGroup:
            return "Clifford group used for RB sequence generation is empty."
        case .insufficientSamplesForFit:
            return "Insufficient survival samples to fit RB decay A·α^m + B."
        case .nonPositiveDecaySignal:
            return "RB decay fit received a non-positive survival signal."
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .invalidQubitCount:
            return "Use RandomizedBenchmarkingOptions.qubitCount of 1 (or supported 2)."
        case .invalidSequenceLength:
            return "Set RandomizedBenchmarkingOptions.sequenceLength to a non-negative integer."
        case .qubitOutOfRange:
            return "Set options.target in 0..<qubitCount."
        case .emptyCliffordGroup:
            return "Use the library Clifford generators for the requested qubitCount."
        case .insufficientSamplesForFit, .nonPositiveDecaySignal:
            return "Collect more sequence lengths / shots with positive survival probability before fitting."
        }
    }
}

extension CrossEntropyBenchmarkError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .invalidQubitCount(let n):
            return "Cross-entropy benchmark qubitCount is invalid (got \(n))."
        case .invalidDepth(let depth):
            return "Cross-entropy benchmark depth is invalid (got \(depth))."
        case .emptyProbabilities:
            return "Ideal probability vector for XEB is empty."
        case .probabilityCountMismatch(let expected, let actual):
            return "XEB probability count \(actual) does not match expected dimension \(expected)."
        case .emptySamples:
            return "XEB sample ensemble is empty."
        case .shotCountMismatch(let expected, let actual):
            return "XEB shot count mismatch (expected \(expected), got \(actual))."
        case .outcomeOutOfRange(let outcome, let dimension):
            return "XEB outcome \(outcome) is outside 0..<\(dimension)."
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .invalidQubitCount, .invalidDepth:
            return "Use CrossEntropyBenchmarkOptions with qubitCount in 1...4 and positive depth."
        case .emptyProbabilities, .probabilityCountMismatch:
            return "Pass a full ideal Born vector of length 2ⁿ matching the circuit width."
        case .emptySamples:
            return "Provide at least one sampled outcome for linear XEB."
        case .shotCountMismatch:
            return "Align the declared shot budget with the length of the outcome sample list."
        case .outcomeOutOfRange:
            return "Encode outcomes as computational-basis indices in 0..<2ⁿ."
        }
    }
}

extension ShorClassicalError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .invalidModulus(let modulus):
            return "Shor classical modulus must be > 1 (got \(modulus))."
        case .baseNotCoprimeToModulus(let base, let modulus):
            return "Base \(base) is not coprime to modulus \(modulus)."
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .invalidModulus:
            return "Pass a modulus N > 1 to ShorClassical.multiplicativeOrder / factoring helpers."
        case .baseNotCoprimeToModulus:
            return "Choose a base coprime to N (gcd(base, N) == 1), or factor via the discovered gcd directly."
        }
    }
}
