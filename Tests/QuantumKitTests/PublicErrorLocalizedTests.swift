import XCTest
@testable import QuantumKit

/// H5: public errors expose non-empty LocalizedError text with actionable remediation keywords.
extension QuantumKitTests {

    func testPublicErrorLocalizedDescriptionsAreActionable() {
        assertActionable(
            QuantumEngineError.deviceNotFound,
            descriptionKeywords: ["Metal"],
            suggestionKeywords: ["SimulationDevicePreference.cpu", "Device-free"]
        )
        assertActionable(
            QuantumEngineError.localizedNoiseRequiresDensityMatrixBackend,
            descriptionKeywords: ["Localized", "noise"],
            suggestionKeywords: ["density-matrix"]
        )
        assertActionable(
            QuantumEngineError.unsupportedGateEncoding(.t(target: 0)),
            descriptionKeywords: ["Unsupported", "gate"],
            suggestionKeywords: ["Transpile", "decompose"]
        )

        assertActionable(
            DensityMatrixEngineError.deviceNotFound,
            descriptionKeywords: ["Metal"],
            suggestionKeywords: ["SimulationDevicePreference.cpu", "Device-free"]
        )

        assertActionable(
            StateVectorError.qubitCountExceedsLimit(max: 28, requested: 30),
            descriptionKeywords: ["exceeds"],
            suggestionKeywords: ["maxPeakMemoryBytes"]
        )
        assertActionable(
            DensityMatrixError.qubitCountExceedsLimit(max: 14, requested: 16),
            descriptionKeywords: ["exceeds"],
            suggestionKeywords: ["trajectory", "maxPeakMemoryBytes"]
        )

        assertActionable(
            SimulationPolicyError.estimatedMemoryExceedsBudget(estimated: 1_000_000, budget: 64),
            descriptionKeywords: ["maxPeakMemoryBytes"],
            suggestionKeywords: ["Increase", "maxPeakMemoryBytes"]
        )
        assertActionable(
            SimulationPolicyError.memoryBudgetRequiresQubitCount,
            descriptionKeywords: ["qubitCount"],
            suggestionKeywords: ["qubitCount"]
        )
        assertActionable(
            SimulationPolicyError.densityMatrixRequiredButTooWide(requested: 20, max: 14),
            descriptionKeywords: ["Density-matrix"],
            suggestionKeywords: ["densityMatrixQubitLimit", "supportsTrajectorySimulation"]
        )

        assertActionable(
            CPUEngineError.localizedNoiseRequiresDensityMatrixBackend,
            descriptionKeywords: ["Localized"],
            suggestionKeywords: ["density-matrix"]
        )
        assertActionable(
            TrajectoryBackendError.shotsRequired,
            descriptionKeywords: ["shots"],
            suggestionKeywords: ["averageProbabilities", "circuit:trajectories:seed:noise:"]
        )

        assertActionable(
            EstimatorError.invalidShotCount(0),
            descriptionKeywords: ["shot"],
            suggestionKeywords: ["shots"]
        )
        assertActionable(
            EstimatorError.shotsRequiredForTrajectory,
            descriptionKeywords: ["Trajectory"],
            suggestionKeywords: ["shots"]
        )
        assertActionable(
            SamplerError.unsupportedBackend,
            descriptionKeywords: ["Sampler"],
            suggestionKeywords: ["statevector"]
        )
        assertActionable(
            QuantumMeasurementError.invalidShotCount(-1),
            descriptionKeywords: ["shot"],
            suggestionKeywords: ["shots"]
        )

        assertActionable(
            TranspilerError.invalidCouplingMap(reason: "empty edge list"),
            descriptionKeywords: ["coupling"],
            suggestionKeywords: ["CouplingMap"]
        )
        assertActionable(
            TranspilerError.circuitWiderThanDevice(circuitQubits: 5, deviceQubits: 3),
            descriptionKeywords: ["width"],
            suggestionKeywords: ["coupling"]
        )
        assertActionable(
            TranspilerError.qubitsNotConnected(0, 2),
            descriptionKeywords: ["connected"],
            suggestionKeywords: ["routing"]
        )
        assertActionable(
            CompilerPassRegistryError.emptyID,
            descriptionKeywords: ["non-empty"],
            suggestionKeywords: ["reverse-DNS"]
        )
        assertActionable(
            CompilerPassRegistryError.duplicateID("x"),
            descriptionKeywords: ["already registered"],
            suggestionKeywords: ["registerReplacing"]
        )
        assertActionable(
            CompilerPassRegistryError.unknownID("missing"),
            descriptionKeywords: ["No CompilerPassFactory"],
            suggestionKeywords: ["Register"]
        )

        assertActionable(
            PECError.incompatibleWithZNE,
            descriptionKeywords: ["PEC", "ZNE"],
            suggestionKeywords: ["Disable"]
        )
        assertActionable(
            PECError.circuitSamplesExceedShots(samples: 100, shots: 10),
            descriptionKeywords: ["circuitSamples"],
            suggestionKeywords: ["shots"]
        )
        assertActionable(
            ZNEError.missingGlobalDepolarizing,
            descriptionKeywords: ["depolarizing"],
            suggestionKeywords: ["depolarizingProbability"]
        )
        assertActionable(
            ZNEError.insufficientScaleFactors(1),
            descriptionKeywords: ["scale"],
            suggestionKeywords: ["scale factors"]
        )

        assertActionable(
            MPSError.noiseNotSupported,
            descriptionKeywords: ["noise"],
            suggestionKeywords: ["density-matrix"]
        )
        assertActionable(
            MPSError.qubitCountMismatch(circuit: 2, state: 1),
            descriptionKeywords: ["width"],
            suggestionKeywords: ["qubitCount"]
        )
        assertActionable(
            MPSError.svdFailed(info: 1),
            descriptionKeywords: ["SVD"],
            suggestionKeywords: ["finite", "statevector", "LAPACK"]
        )
        assertActionable(
            StabilizerError.nonCliffordGate(.t(target: 0)),
            descriptionKeywords: ["Clifford"],
            suggestionKeywords: ["Clifford"]
        )
        assertActionable(
            StabilizerError.qubitCountMismatch(circuit: 3, tableau: 1),
            descriptionKeywords: ["width"],
            suggestionKeywords: ["qubitCount"]
        )
        assertActionable(
            SimulationPrecisionError.metalFloat64Unsupported,
            descriptionKeywords: ["Float64"],
            suggestionKeywords: ["cpu"]
        )

        // H5 residuals — at least one case per remaining public Error enum
        assertNonEmptyDescription(QuantumCircuitError.invalidQubitCount(0))
        assertActionable(
            QuantumCircuitError.qubitIndexOutOfBounds(index: 3, qubitCount: 2),
            descriptionKeywords: ["out of bounds"],
            suggestionKeywords: ["0..<qubitCount"]
        )
        assertActionable(
            QuantumCircuitError.maxLoopIterationsExceeded(maxIterations: 5),
            descriptionKeywords: ["maxIterations", "5"],
            suggestionKeywords: ["maxIterations"]
        )

        assertNonEmptyDescription(CircuitIRError.unsupportedSchemaVersion(found: 99, supported: CircuitIRSchema.current))
        assertActionable(
            CircuitIRError.metadataLengthMismatch(metadataCount: 1, gateCount: 2),
            descriptionKeywords: ["instructionMetadata"],
            suggestionKeywords: ["gates"]
        )
        assertActionable(
            CircuitIRError.controlFlowNotSerialized(op: "while_c"),
            descriptionKeywords: ["while_c", "cannot be encoded"],
            suggestionKeywords: ["while_c"]
        )

        assertNonEmptyDescription(ParameterBindingError.missingBinding(for: "theta"))
        assertActionable(
            ParameterBindingError.circuitContainsUnboundParameters(["gamma"]),
            descriptionKeywords: ["unbound"],
            suggestionKeywords: ["QuantumParameter"]
        )

        assertActionable(
            BatchExecutionError.invalidBatchSize(0),
            descriptionKeywords: ["capacity"],
            suggestionKeywords: ["StateVectorBatch"]
        )

        assertNonEmptyDescription(
            StateVectorInitializationError.amplitudeCountMismatch(expected: 4, actual: 2)
        )
        assertActionable(
            StateVectorInitializationError.nonUnitNorm(squaredNorm: 0.5),
            descriptionKeywords: ["norm"],
            suggestionKeywords: ["Normalize"]
        )

        assertNonEmptyDescription(
            DensityMatrixInitializationError.amplitudeCountMismatch(expected: 4, actual: 1)
        )
        assertActionable(
            DensityMatrixInitializationError.nonUnitNorm(squaredNorm: 2.0),
            descriptionKeywords: ["norm"],
            suggestionKeywords: ["Normalize"]
        )

        assertNonEmptyDescription(QuantumChannelError.emptyKrausSet)
        assertActionable(
            QuantumChannelError.pauliProbabilitiesExceedOne(sum: 1.5),
            descriptionKeywords: ["Pauli"],
            suggestionKeywords: ["makePauliChannel"]
        )
        assertActionable(
            QuantumChannelError.emptyKrausSet,
            descriptionKeywords: ["Kraus"],
            suggestionKeywords: ["fromKraus1Q"]
        )

        assertNonEmptyDescription(ReadoutConfusionError.emptyMatrix)
        assertActionable(
            ReadoutConfusionError.nonStochasticRow(row: 0),
            descriptionKeywords: ["stochastic"],
            suggestionKeywords: ["probability"]
        )

        assertNonEmptyDescription(ReadoutMitigationError.singularConfusionMatrix)
        assertActionable(
            ReadoutMitigationError.qubitCountMismatch(histogram: 2, matrix: 1),
            descriptionKeywords: ["mismatch"],
            suggestionKeywords: ["ReadoutMitigation.apply"]
        )

        assertNonEmptyDescription(PostSelectionError.missingShotCounts)
        assertActionable(
            PostSelectionError.emptyKeepSet(discardedShots: 100),
            descriptionKeywords: ["zero shots"],
            suggestionKeywords: ["EmptyKeepSetPolicy"]
        )

        assertNonEmptyDescription(GradientCalculatorError.noDifferentiableParameters)
        assertActionable(
            GradientCalculatorError.hessianRequiresParameterShift,
            descriptionKeywords: ["Hessian"],
            suggestionKeywords: ["parameterShift"]
        )
        assertActionable(
            GradientCalculatorError.adjointRequiresStatevectorBackend,
            descriptionKeywords: ["statevector"],
            suggestionKeywords: ["StatevectorBackend"]
        )

        assertNonEmptyDescription(
            CheckpointError.instructionIndexOutOfBounds(index: 5, gateCount: 3)
        )
        assertActionable(
            CheckpointError.qubitCountMismatch(expected: 2, actual: 3),
            descriptionKeywords: ["qubitCount"],
            suggestionKeywords: ["matching"]
        )

        assertActionable(
            CircuitExecutionCancellationError.cancelled,
            descriptionKeywords: ["cancelled"],
            suggestionKeywords: ["fresh", "StateVector"]
        )

        assertNonEmptyDescription(DAGCircuitError.cycleDetected)
        assertActionable(
            DAGCircuitError.invalidCircuit(reason: "qubitCount must be positive"),
            descriptionKeywords: ["DAGCircuit"],
            suggestionKeywords: ["qubitCount"]
        )

        assertNonEmptyDescription(
            AncillaAllocationError.insufficientAncillas(required: 2, available: 0)
        )
        assertActionable(
            AncillaAllocationError.ancillaAllocationDisabled(strategy: .vChainAncilla),
            descriptionKeywords: ["Ancilla"],
            suggestionKeywords: ["TranspileOptions"]
        )

        assertNonEmptyDescription(QubitBitOrderingError.invalidBitCharacter("2"))
        assertActionable(
            QubitBitOrderingError.invalidBitstringLength(expected: 3, actual: 2),
            descriptionKeywords: ["Bitstring"],
            suggestionKeywords: ["fromBitstring"]
        )

        assertNonEmptyDescription(OperatorError.invalidPauliLabel("Q0"))
        assertActionable(
            OperatorError.duplicateQubitInTerm(qubit: 0, label: "Z0 X0"),
            descriptionKeywords: ["Duplicate"],
            suggestionKeywords: ["PauliTerm"]
        )

        assertNonEmptyDescription(ClassicalMemoryError.invalidBitCount(0))
        assertActionable(
            ClassicalMemoryError.bitOffsetOutOfBounds(register: 0, offset: 3, width: 2),
            descriptionKeywords: ["offset"],
            suggestionKeywords: ["registerWidths"]
        )

        assertNonEmptyDescription(TrotterError.invalidStepCount(0))
        assertActionable(
            TrotterError.qubitIndexOutOfRange(qubit: 5, qubitCount: 2),
            descriptionKeywords: ["Pauli"],
            suggestionKeywords: ["qubitCount"]
        )

        assertNonEmptyDescription(VariationalAlgorithmError.emptyProblem)
        assertActionable(
            VariationalAlgorithmError.parameterCountMismatch(expected: 4, actual: 2),
            descriptionKeywords: ["parameter"],
            suggestionKeywords: ["QAOA"]
        )

        assertNonEmptyDescription(QuantumVolumeError.emptyProbabilities)
        assertActionable(
            QuantumVolumeError.invalidQubitCount(0),
            descriptionKeywords: ["qubitCount"],
            suggestionKeywords: ["QuantumVolume"]
        )

        assertNonEmptyDescription(RandomizedBenchmarkingError.emptyCliffordGroup)
        assertActionable(
            RandomizedBenchmarkingError.insufficientSamplesForFit,
            descriptionKeywords: ["samples"],
            suggestionKeywords: ["survival"]
        )

        assertNonEmptyDescription(CrossEntropyBenchmarkError.emptySamples)
        assertActionable(
            CrossEntropyBenchmarkError.invalidQubitCount(8),
            descriptionKeywords: ["qubitCount"],
            suggestionKeywords: ["CrossEntropyBenchmarkOptions"]
        )

        assertNonEmptyDescription(ShorClassicalError.invalidModulus(1))
        assertActionable(
            ShorClassicalError.baseNotCoprimeToModulus(base: 2, modulus: 4),
            descriptionKeywords: ["coprime"],
            suggestionKeywords: ["gcd"]
        )
    }

    func testPublicErrorEquatableStillHoldsForSampleCases() {
        XCTAssertEqual(
            SimulationPolicyError.memoryBudgetRequiresQubitCount,
            SimulationPolicyError.memoryBudgetRequiresQubitCount
        )
        XCTAssertEqual(
            TranspilerError.qubitsNotConnected(1, 2),
            TranspilerError.qubitsNotConnected(1, 2)
        )
        XCTAssertEqual(PECError.incompatibleWithZNE, PECError.incompatibleWithZNE)
        XCTAssertEqual(ZNEError.singularLinearFit, ZNEError.singularLinearFit)
        XCTAssertEqual(EstimatorError.unsupportedBackend, EstimatorError.unsupportedBackend)
        XCTAssertEqual(CPUEngineError.zeroStateNorm, CPUEngineError.zeroStateNorm)
        XCTAssertEqual(MPSError.noiseNotSupported, MPSError.noiseNotSupported)
        XCTAssertEqual(StabilizerError.noiseNotSupported, StabilizerError.noiseNotSupported)
        XCTAssertEqual(QuantumCircuitError.circuitNotUnitary, QuantumCircuitError.circuitNotUnitary)
        XCTAssertEqual(
            ParameterBindingError.missingBinding(for: "θ"),
            ParameterBindingError.missingBinding(for: "θ")
        )
        XCTAssertEqual(PostSelectionError.missingShotCounts, PostSelectionError.missingShotCounts)
        XCTAssertEqual(
            CircuitExecutionCancellationError.cancelled,
            CircuitExecutionCancellationError.cancelled
        )
        XCTAssertEqual(ShorClassicalError.invalidModulus(0), ShorClassicalError.invalidModulus(0))
        XCTAssertEqual(QuantumKitInfo.version, "0.2.0")
    }

    /// `densityMatrixRequiredButTooWide` is thrown when DM is required but too wide *and*
    /// trajectory cannot be selected (incompatible noise and/or preferTrajectory off).
    /// Suggestion must not blindly tell users to "enable trajectory".
    func testDensityMatrixRequiredButTooWideSuggestionDoesNotBlindlyEnableTrajectory() {
        let error = SimulationPolicyError.densityMatrixRequiredButTooWide(requested: 20, max: 14)
        let suggestion = error.recoverySuggestion ?? ""
        XCTAssertFalse(suggestion.isEmpty)
        XCTAssertTrue(
            suggestion.localizedCaseInsensitiveContains("densityMatrixQubitLimit"),
            "expected DM limit remediation, got: \(suggestion)"
        )
        XCTAssertFalse(
            suggestion.localizedCaseInsensitiveContains("Enable trajectory"),
            "must not unconditionally advise enabling trajectory: \(suggestion)"
        )
        XCTAssertTrue(
            suggestion.localizedCaseInsensitiveContains("supportsTrajectorySimulation")
                || suggestion.localizedCaseInsensitiveContains("preferTrajectoryWhenDensityMatrixTooWide"),
            "should qualify when trajectory is an alternative: \(suggestion)"
        )
        XCTAssertTrue(
            suggestion.localizedCaseInsensitiveContains("will not help")
                || suggestion.localizedCaseInsensitiveContains("otherwise simplify"),
            "should state trajectory may not apply: \(suggestion)"
        )
    }

    private func assertNonEmptyDescription(
        _ error: LocalizedError,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let description = error.errorDescription ?? ""
        XCTAssertFalse(
            description.isEmpty,
            "errorDescription should be non-empty for \(error)",
            file: file,
            line: line
        )
    }

    private func assertActionable(
        _ error: LocalizedError,
        descriptionKeywords: [String],
        suggestionKeywords: [String],
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let description = error.errorDescription ?? ""
        XCTAssertFalse(description.isEmpty, "errorDescription should be non-empty for \(error)", file: file, line: line)
        for keyword in descriptionKeywords {
            XCTAssertTrue(
                description.localizedCaseInsensitiveContains(keyword),
                "errorDescription '\(description)' should mention '\(keyword)'",
                file: file,
                line: line
            )
        }

        let suggestion = error.recoverySuggestion ?? ""
        XCTAssertFalse(suggestion.isEmpty, "recoverySuggestion should be non-empty for \(error)", file: file, line: line)
        for keyword in suggestionKeywords {
            XCTAssertTrue(
                suggestion.localizedCaseInsensitiveContains(keyword),
                "recoverySuggestion '\(suggestion)' should mention '\(keyword)'",
                file: file,
                line: line
            )
        }
    }
}
