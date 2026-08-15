import Foundation

public enum EstimatorError: Error, Equatable {
    case unsupportedBackend
    case invalidShotCount(Int)
    case invalidPrecision(QFloat)
    /// Trajectory expectations are ensemble estimates; set ``EstimatorOptions/shots`` or ``QuantumRunOptions/shots``.
    case shotsRequiredForTrajectory
}

/// Options for ``Estimator`` beyond the shared ``QuantumRunOptions``.
///
/// ## Shot budget
/// When `shots` or `precision` is set, the estimator uses Pauli measurement sampling
/// instead of exact analytic expectations. Resolution order (see also ``ShotBudget``):
/// 1. ``shots`` (explicit)
/// 2. else ``precision`` → `shots ≈ ⌈1/ε²⌉`
/// 3. else ``QuantumRunOptions/shots``
/// 4. else exact ⟨H⟩ / Tr(ρH)
///
/// ## Resilience
/// ``resilience`` (when enabled) applies host-side readout mitigation to each shot
/// ensemble before parity moments, optionally **zero-noise extrapolation** (``ZNEOptions``),
/// and/or **PEC lite** (``PECOptions``) on the shot path. ZNE and PEC cannot be combined
/// in this MVP. If ``resilience`` is ``disabled``, the estimator falls back to
/// ``QuantumRunOptions/resilience``. Exact paths ignore resilience (including ZNE/PEC).
/// Resolved resilience is included in ``EstimatorResult/metadata`` ``pipelineHash``.
///
/// ## QWC grouping
/// ``groupCommutingPaulis`` (default `true`) partitions non-identity terms into
/// qubit-wise commuting groups so each group shares one shot ensemble. Set `false`
/// for legacy per-term ensembles (seed-stable vs pre-grouping Estimator schedules).
public struct EstimatorOptions: Sendable, Equatable {
    /// Explicit shot count for Pauli sampling. Wins over ``precision`` when both are set.
    public var shots: Int?
    /// Target absolute standard-error scale; translated to `shots ≈ 1/precision²` when
    /// `shots` is `nil`.
    public var precision: QFloat?
    /// Opt-in resilience knobs (default disabled). See ``ResilienceOptions``.
    public var resilience: ResilienceOptions
    /// When `true` (default), partition into QWC groups. When `false`, each non-identity
    /// term gets its own ensemble (legacy schedule).
    public var groupCommutingPaulis: Bool

    public init(
        shots: Int? = nil,
        precision: QFloat? = nil,
        resilience: ResilienceOptions = .disabled,
        groupCommutingPaulis: Bool = true
    ) {
        self.shots = shots
        self.precision = precision
        self.resilience = resilience
        self.groupCommutingPaulis = groupCommutingPaulis
    }

    /// Exact analytic path (default).
    public static let exact = EstimatorOptions()

    /// Shot / precision pair as a ``ShotBudget`` (same fields; for shared documentation).
    public var shotBudget: ShotBudget {
        ShotBudget(shots: shots, precision: precision)
    }

    /// Resolves the shot count to use, or `nil` for the exact estimator.
    public func resolvedShots() throws -> Int? {
        try shotBudget.resolvedShots()
    }

    /// Estimator resilience if set; otherwise the run-options resilience.
    ///
    /// When Estimator resilience is enabled (readout / active ZNE / PEC), missing
    /// ``ResilienceOptions/readoutMitigation`` is inherited from ``QuantumRunOptions/resilience``
    /// so enabling PEC or ZNE alone does not silently drop run-options readout correction.
    func resolvedResilience(runOptions: QuantumRunOptions) -> ResilienceOptions {
        guard resilience.isEnabled else { return runOptions.resilience }
        guard resilience.readoutMitigation == nil,
              let runReadout = runOptions.resilience.readoutMitigation else {
            return resilience
        }
        return ResilienceOptions(
            readoutMitigation: runReadout,
            zne: resilience.zne,
            pec: resilience.pec
        )
    }

    /// Tokens folded into ``PipelineFingerprint`` for Estimator jobs.
    var fingerprintExtra: [String] {
        ["qwc:\(groupCommutingPaulis ? 1 : 0)"]
    }
}

/// Result of an ``Estimator`` run: the Hamiltonian expectation value plus execution metadata.
public struct EstimatorResult: Sendable, Equatable {
    public let value: QFloat
    public let metadata: QuantumResultMetadata
    /// Shot count when sampling was used; `nil` for exact expectations.
    public let shots: Int?
    /// Estimated standard error of the mean for shot-based runs (`≈ √(Var̂/shots)`).
    /// `Var̂` includes within-QWC-group covariances; see ``Estimator`` shot-path docs.
    /// When ZNE is enabled this is the SE from the **nominal** (`λ` closest to 1, else first)
    /// scale evaluation — not an extrapolated uncertainty.
    /// When PEC is enabled this is an approximate SE from the signed sample variance
    /// scaled by `Γ` (see ``PECMetadata/shotMultiplier``).
    public let standardError: QFloat?
    /// Present when shot-path ``ResilienceOptions/zne`` ran; `nil` otherwise.
    public let zne: ZNEExtrapolationMetadata?
    /// Present when shot-path ``ResilienceOptions/pec`` ran; `nil` otherwise.
    public let pec: PECMetadata?

    public init(
        value: QFloat,
        metadata: QuantumResultMetadata,
        shots: Int? = nil,
        standardError: QFloat? = nil,
        zne: ZNEExtrapolationMetadata? = nil,
        pec: PECMetadata? = nil
    ) {
        self.value = value
        self.metadata = metadata
        self.shots = shots
        self.standardError = standardError
        self.zne = zne
        self.pec = pec
    }
}

/// High-level primitive for evaluating ⟨ψ|H|ψ⟩ (or Tr(ρH)) after circuit evolution.
///
/// Exact path: GPU Pauli kernels on ``StatevectorBackend`` / ``DensityMatrixBackend``, and
/// host Double paths on ``CPUStatevectorBackend`` / ``CPUDensityMatrixBackend``. Exact
/// semantics are unchanged: one evolve of the user circuit, then Σ cᵢ ⟨Pᵢ⟩ / Tr(ρPᵢ).
///
/// Shot path (``EstimatorOptions/shots`` / ``precision``): Pauli-basis sampling on
/// statevector, density-matrix (prepared-ρ batching), CPU, and trajectory backends.
/// Non-identity terms are partitioned into **qubit-wise commuting (QWC)** groups
/// (``PauliCommutingGroups``) when ``EstimatorOptions/groupCommutingPaulis`` is `true`
/// (default); each group shares one basis-change circuit and one shot ensemble.
/// Set `groupCommutingPaulis` to `false` for legacy per-term ensembles.
///
/// **Reproducibility:** with default grouping, multi-term Hamiltonians no longer draw an
/// independent ensemble per term. Seeded shot values for commuting multi-term `H`
/// therefore differ from pre-grouping Estimator runs with the same seed, even when
/// resilience is disabled. Single-term, exact, and `groupCommutingPaulis == false` paths
/// match the legacy schedule. Expectation remains unbiased; only the RNG schedule /
/// shared-shot allocation changes.
///
/// **Standard error:** ``EstimatorResult/standardError`` is `√(Var̂/shots)` where `Var̂`
/// sums **per-group** sample variances of `Σ cᵢ Pᵢ` on that group's shared ensemble
/// (covariances inside a QWC group are included). Distinct groups use independent
/// ensembles, so their variances add. Trajectory requires shots (no exact mixed-state path).
public struct Estimator: Sendable {
    public init() {}

    public func run(
        circuit: QuantumCircuit,
        hamiltonian: Hamiltonian,
        backend: any QuantumBackend,
        options: QuantumRunOptions = QuantumRunOptions(),
        estimatorOptions: EstimatorOptions = .exact
    ) throws -> EstimatorResult {
        try SimulationProfiling.usingRecorder(for: options) {
            let started = DispatchTime.now()
            let resolvedShots = try estimatorOptions.resolvedShots() ?? options.shots
            // Nested under Gradient (etc.): exact path still records user-circuit gates on the
            // outer recorder; shot path suppresses gate samples (basis-changed circuits).
            // Do not emit an orphan "estimate" phase or a mid-run finishProfile snapshot.
            let nested = SimulationProfiling.isNestedUnderOuterRecorder

            if let shots = resolvedShots {
                return try estimateShots(
                    circuit: circuit,
                    hamiltonian: hamiltonian,
                    backend: backend,
                    options: options,
                    estimatorOptions: estimatorOptions,
                    shots: shots,
                    started: started,
                    nestedUnderOuterRecorder: nested
                )
            }

            let estimateBody = { () -> (QFloat, QuantumSimulationMethod, String?) in
                if let statevectorBackend = backend as? StatevectorBackend {
                    let value = try estimateExact(
                        circuit: circuit,
                        hamiltonian: hamiltonian,
                        engine: statevectorBackend.engine,
                        options: options
                    )
                    return (value, .statevector, MetalRuntime.deviceName)
                }
                if let densityBackend = backend as? DensityMatrixBackend {
                    let value = try estimateExact(
                        circuit: circuit,
                        hamiltonian: hamiltonian,
                        engine: densityBackend.engine,
                        options: options
                    )
                    return (value, .densityMatrix, MetalRuntime.deviceName)
                }
                if let cpuSV = backend as? CPUStatevectorBackend {
                    let value = try estimateExactCPU(
                        circuit: circuit,
                        hamiltonian: hamiltonian,
                        engine: cpuSV.engine,
                        options: options
                    )
                    return (value, .statevector, "CPU")
                }
                if let cpuDM = backend as? CPUDensityMatrixBackend {
                    let value = try estimateExactCPU(
                        circuit: circuit,
                        hamiltonian: hamiltonian,
                        engine: cpuDM.engine,
                        options: options
                    )
                    return (value, .densityMatrix, "CPU")
                }
                if backend is TrajectoryBackend {
                    throw EstimatorError.shotsRequiredForTrajectory
                }
                throw EstimatorError.unsupportedBackend
            }
            let boxed = try nested
                ? estimateBody()
                : SimulationProfiling.timePhase("estimate", estimateBody)

            let elapsed = DispatchTime.now().uptimeNanoseconds - started.uptimeNanoseconds
            let fingerprintOptions = PipelineFingerprint.optionsForEstimatorFingerprint(
                runOptions: options,
                resolvedShots: nil,
                resolvedResilience: .disabled
            )
            let metadata = QuantumResultMetadata(
                method: boxed.1,
                seed: options.seed,
                deviceName: boxed.2,
                wallClockNanoseconds: elapsed,
                qubitCount: circuit.qubitCount,
                gateCount: circuit.gates.count,
                noiseSnapshot: options.noise,
                pipelineHash: PipelineFingerprint.hash(
                    circuit: circuit,
                    method: boxed.1,
                    options: fingerprintOptions,
                    extra: estimatorOptions.fingerprintExtra
                ),
                profile: nested
                    ? nil
                    : SimulationProfiling.finishProfile(
                        options: options,
                        circuit: circuit,
                        method: boxed.1,
                        isCPU: boxed.2 == "CPU",
                        elapsed: elapsed
                    )
            )

            return EstimatorResult(value: boxed.0, metadata: metadata)
        }
    }

    private func estimateExact(
        circuit: QuantumCircuit,
        hamiltonian: Hamiltonian,
        engine: QuantumEngine,
        options: QuantumRunOptions
    ) throws -> QFloat {
        let state = try StateVector(qubitCount: circuit.qubitCount)
        var rng = makePrimitiveRNG(seed: options.seed)
        _ = try engine.executeRNG(circuit, on: state, rng: &rng, noise: options.noise)
        return try hamiltonian.expectation(state: state, engine: engine)
    }

    private func estimateExact(
        circuit: QuantumCircuit,
        hamiltonian: Hamiltonian,
        engine: DensityMatrixEngine,
        options: QuantumRunOptions
    ) throws -> QFloat {
        let density = try DensityMatrix(qubitCount: circuit.qubitCount)
        var rng = makePrimitiveRNG(seed: options.seed)
        _ = try engine.executeRNG(circuit, on: density, rng: &rng, noise: options.noise)
        return try hamiltonian.expectation(density: density, engine: engine)
    }

    private func estimateExactCPU(
        circuit: QuantumCircuit,
        hamiltonian: Hamiltonian,
        engine: CPUStatevectorEngine,
        options: QuantumRunOptions
    ) throws -> QFloat {
        let state = try CPUStateVector(qubitCount: circuit.qubitCount)
        var rng = makePrimitiveRNG(seed: options.seed)
        _ = try engine.executeRNG(circuit, on: state, rng: &rng, noise: options.noise)
        return try hamiltonian.expectation(state: state)
    }

    private func estimateExactCPU(
        circuit: QuantumCircuit,
        hamiltonian: Hamiltonian,
        engine: CPUDensityMatrixEngine,
        options: QuantumRunOptions
    ) throws -> QFloat {
        let density = try CPUDensityMatrix(qubitCount: circuit.qubitCount)
        var rng = makePrimitiveRNG(seed: options.seed)
        _ = try engine.executeRNG(circuit, on: density, rng: &rng, noise: options.noise)
        return try hamiltonian.expectation(density: density)
    }

    private func estimateShots(
        circuit: QuantumCircuit,
        hamiltonian: Hamiltonian,
        backend: any QuantumBackend,
        options: QuantumRunOptions,
        estimatorOptions: EstimatorOptions,
        shots: Int,
        started: DispatchTime,
        nestedUnderOuterRecorder: Bool
    ) throws -> EstimatorResult {
        guard shots > 0 else { throw EstimatorError.invalidShotCount(shots) }
        let resilience = estimatorOptions.resolvedResilience(runOptions: options)
        let activeZNE = resilience.activeZNE

        if activeZNE != nil, resilience.pec != nil {
            throw PECError.incompatibleWithZNE
        }

        if let pecOptions = resilience.pec {
            return try estimateShotsWithPEC(
                circuit: circuit,
                hamiltonian: hamiltonian,
                backend: backend,
                options: options,
                estimatorOptions: estimatorOptions,
                shots: shots,
                started: started,
                nestedUnderOuterRecorder: nestedUnderOuterRecorder,
                resilience: resilience,
                pecOptions: pecOptions
            )
        }

        if let zneOptions = activeZNE {
            try ZeroNoiseExtrapolation.validate(zneOptions)
            return try estimateShotsWithZNE(
                circuit: circuit,
                hamiltonian: hamiltonian,
                backend: backend,
                options: options,
                estimatorOptions: estimatorOptions,
                shots: shots,
                started: started,
                nestedUnderOuterRecorder: nestedUnderOuterRecorder,
                resilience: resilience,
                zneOptions: zneOptions
            )
        }

        // Strip inactive ZNE/PEC tokens so nested shot paths and fingerprints stay readout-clean.
        let nestedResilience = resilience.readoutOnly
        let boxed = try runShotExpectation(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: backend,
            options: options,
            estimatorOptions: estimatorOptions,
            shots: shots,
            noise: options.noise,
            resilience: nestedResilience,
            nestedUnderOuterRecorder: nestedUnderOuterRecorder
        )

        let standardError = sqrt(boxed.variance / QFloat(shots))
        let elapsed = DispatchTime.now().uptimeNanoseconds - started.uptimeNanoseconds
        let metadata = makeShotMetadata(
            circuit: circuit,
            options: options,
            estimatorOptions: estimatorOptions,
            shots: shots,
            resilience: nestedResilience,
            method: boxed.method,
            deviceName: boxed.deviceName,
            elapsed: elapsed,
            nestedUnderOuterRecorder: nestedUnderOuterRecorder
        )

        return EstimatorResult(
            value: boxed.value,
            metadata: metadata,
            shots: shots,
            standardError: standardError
        )
    }

    private func estimateShotsWithPEC(
        circuit: QuantumCircuit,
        hamiltonian: Hamiltonian,
        backend: any QuantumBackend,
        options: QuantumRunOptions,
        estimatorOptions: EstimatorOptions,
        shots: Int,
        started: DispatchTime,
        nestedUnderOuterRecorder: Bool,
        resilience: ResilienceOptions,
        pecOptions: PECOptions
    ) throws -> EstimatorResult {
        guard let noise = options.noise else { throw PECError.missingNoiseModel }
        let sites = try ProbabilisticErrorCancellation.validatedPECSites(
            circuit: circuit,
            noise: noise
        )
        let qpr = try ProbabilisticErrorCancellation.quasiprobability(
            channel: pecOptions.channel,
            noise: noise
        )
        let gammaPerSite = qpr.gamma
        let siteCount = sites.count
        let gammaTotal = pow(gammaPerSite, QFloat(siteCount))

        let circuitSamples: Int
        if let requested = pecOptions.circuitSamples {
            guard requested > 0 else { throw PECError.invalidCircuitSampleCount(requested) }
            guard requested <= shots else {
                throw PECError.circuitSamplesExceedShots(samples: requested, shots: shots)
            }
            circuitSamples = requested
        } else {
            circuitSamples = shots
        }
        let shotsPerCircuit = max(1, shots / circuitSamples)
        let effectiveShots = circuitSamples * shotsPerCircuit
        let perSampleResilience = resilience.readoutOnly

        var weightedSum: Double = 0
        var weightedSumSq: Double = 0
        var method: QuantumSimulationMethod = .statevector
        var deviceName: String?

        for sampleIndex in 0..<circuitSamples {
            var sampleRNG = makePrimitiveRNG(
                seed: options.seed.map { $0 &+ UInt64(sampleIndex) &+ 0x9E37_79B9 }
            )
            var paulish: [Pauli] = []
            var sign: QFloat = 1
            paulish.reserveCapacity(siteCount)
            for _ in 0..<siteCount {
                let draw = qpr.sample(rng: &sampleRNG)
                paulish.append(draw.pauli)
                sign *= draw.sign
            }

            let mitigated = try ProbabilisticErrorCancellation.circuitByInsertingMitigationPaulis(
                circuit,
                siteGateIndices: sites,
                paulish: paulish
            )

            var sampleOptions = options
            if let seed = options.seed {
                sampleOptions.seed = seed &+ UInt64(sampleIndex)
            }

            let boxed = try runShotExpectation(
                circuit: mitigated,
                hamiltonian: hamiltonian,
                backend: backend,
                options: sampleOptions,
                estimatorOptions: estimatorOptions,
                shots: shotsPerCircuit,
                noise: noise,
                resilience: perSampleResilience,
                nestedUnderOuterRecorder: nestedUnderOuterRecorder
            )
            let signed = Double(sign) * Double(boxed.value)
            weightedSum += signed
            weightedSumSq += signed * signed
            method = boxed.method
            deviceName = boxed.deviceName
        }

        let meanSigned = weightedSum / Double(circuitSamples)
        let extrapolated = QFloat(Double(gammaTotal) * meanSigned)
        let meanSq = weightedSumSq / Double(circuitSamples)
        let varSigned = max(0, meanSq - meanSigned * meanSigned)
        let standardError = QFloat(
            Double(gammaTotal) * sqrt(varSigned / Double(circuitSamples))
        )

        let pecMeta = PECMetadata(
            channel: pecOptions.channel,
            gammaPerSite: gammaPerSite,
            gammaTotal: gammaTotal,
            siteCount: siteCount,
            circuitSamples: circuitSamples
        )

        let elapsed = DispatchTime.now().uptimeNanoseconds - started.uptimeNanoseconds
        let metadata = makeShotMetadata(
            circuit: circuit,
            options: options,
            estimatorOptions: estimatorOptions,
            shots: effectiveShots,
            resilience: resilience,
            method: method,
            deviceName: deviceName,
            elapsed: elapsed,
            nestedUnderOuterRecorder: nestedUnderOuterRecorder
        )

        return EstimatorResult(
            value: extrapolated,
            metadata: metadata,
            shots: effectiveShots,
            standardError: standardError,
            pec: pecMeta
        )
    }

    private func estimateShotsWithZNE(
        circuit: QuantumCircuit,
        hamiltonian: Hamiltonian,
        backend: any QuantumBackend,
        options: QuantumRunOptions,
        estimatorOptions: EstimatorOptions,
        shots: Int,
        started: DispatchTime,
        nestedUnderOuterRecorder: Bool,
        resilience: ResilienceOptions,
        zneOptions: ZNEOptions
    ) throws -> EstimatorResult {
        try ZeroNoiseExtrapolation.validate(zneOptions)
        guard let noise = options.noise, noise.appliesDepolarizing else {
            throw ZNEError.missingGlobalDepolarizing
        }
        let perScaleResilience = resilience.readoutOnly

        var valuesAtScale: [QFloat] = []
        valuesAtScale.reserveCapacity(zneOptions.scaleFactors.count)
        var varianceAtScale: [QFloat] = []
        varianceAtScale.reserveCapacity(zneOptions.scaleFactors.count)
        var method: QuantumSimulationMethod = .statevector
        var deviceName: String?

        for (scaleIndex, scale) in zneOptions.scaleFactors.enumerated() {
            let scaledNoise = options.noise.map { $0.scalingGlobalDepolarizing(by: scale) }
            var scaleOptions = options
            // Independent ensemble per scale; preserve seeded reproducibility.
            if let seed = options.seed {
                scaleOptions.seed = seed &+ UInt64(scaleIndex)
            }
            scaleOptions.noise = scaledNoise

            let boxed = try runShotExpectation(
                circuit: circuit,
                hamiltonian: hamiltonian,
                backend: backend,
                options: scaleOptions,
                estimatorOptions: estimatorOptions,
                shots: shots,
                noise: scaledNoise,
                resilience: perScaleResilience,
                nestedUnderOuterRecorder: nestedUnderOuterRecorder
            )
            valuesAtScale.append(boxed.value)
            varianceAtScale.append(boxed.variance)
            method = boxed.method
            deviceName = boxed.deviceName
        }

        let extrapolated = try ZeroNoiseExtrapolation.extrapolate(
            options: zneOptions,
            valuesAtScale: valuesAtScale
        )
        let zneMeta = ZNEExtrapolationMetadata(
            extrapolator: zneOptions.extrapolator,
            scaleFactors: zneOptions.scaleFactors,
            valuesAtScale: valuesAtScale,
            extrapolatedValue: extrapolated
        )

        let seIndex = zneOptions.scaleFactors.firstIndex(where: { abs($0 - 1) < 1e-6 }) ?? 0
        let standardError = sqrt(varianceAtScale[seIndex] / QFloat(shots))
        let elapsed = DispatchTime.now().uptimeNanoseconds - started.uptimeNanoseconds
        let metadata = makeShotMetadata(
            circuit: circuit,
            options: options,
            estimatorOptions: estimatorOptions,
            shots: shots,
            resilience: resilience,
            method: method,
            deviceName: deviceName,
            elapsed: elapsed,
            nestedUnderOuterRecorder: nestedUnderOuterRecorder
        )

        return EstimatorResult(
            value: extrapolated,
            metadata: metadata,
            shots: shots,
            standardError: standardError,
            zne: zneMeta
        )
    }

    private func makeShotMetadata(
        circuit: QuantumCircuit,
        options: QuantumRunOptions,
        estimatorOptions: EstimatorOptions,
        shots: Int,
        resilience: ResilienceOptions,
        method: QuantumSimulationMethod,
        deviceName: String?,
        elapsed: UInt64,
        nestedUnderOuterRecorder: Bool
    ) -> QuantumResultMetadata {
        let fingerprintOptions = PipelineFingerprint.optionsForEstimatorFingerprint(
            runOptions: options,
            resolvedShots: shots,
            resolvedResilience: resilience
        )
        return QuantumResultMetadata(
            method: method,
            seed: options.seed,
            deviceName: deviceName,
            wallClockNanoseconds: elapsed,
            qubitCount: circuit.qubitCount,
            gateCount: circuit.gates.count,
            noiseSnapshot: options.noise,
            pipelineHash: PipelineFingerprint.hash(
                circuit: circuit,
                method: method,
                options: fingerprintOptions,
                extra: estimatorOptions.fingerprintExtra
            ),
            profile: nestedUnderOuterRecorder
                ? nil
                : SimulationProfiling.finishProfile(
                    options: options,
                    circuit: circuit,
                    method: method,
                    isCPU: deviceName == "CPU",
                    elapsed: elapsed,
                    effectiveShots: shots,
                    includeGateTimings: false
                )
        )
    }

    private struct ShotExpectationBox: Sendable {
        let value: QFloat
        let variance: QFloat
        let method: QuantumSimulationMethod
        let deviceName: String?
    }

    private func runShotExpectation(
        circuit: QuantumCircuit,
        hamiltonian: Hamiltonian,
        backend: any QuantumBackend,
        options: QuantumRunOptions,
        estimatorOptions: EstimatorOptions,
        shots: Int,
        noise: NoiseModel?,
        resilience: ResilienceOptions,
        nestedUnderOuterRecorder: Bool
    ) throws -> ShotExpectationBox {
        let shotsBody = { () -> ShotExpectationBox in
            var rng = makePrimitiveRNG(seed: options.seed)
            let partition: PauliCommutingGroups.Partition
            if estimatorOptions.groupCommutingPaulis {
                partition = PauliCommutingGroups.partition(hamiltonian)
            } else {
                partition = PauliCommutingGroups.partitionSingletons(hamiltonian)
            }
            var total: QFloat = partition.identityContribution
            var varianceAccumulator: QFloat = 0
            let method: QuantumSimulationMethod
            let deviceName: String?

            if let statevectorBackend = backend as? StatevectorBackend {
                method = .statevector
                deviceName = MetalRuntime.deviceName
                for group in partition.groups {
                    let groupMoments = try PauliShotEstimator.estimateGroup(
                        circuit: circuit,
                        group: group,
                        engine: statevectorBackend.engine,
                        shots: shots,
                        rng: &rng,
                        noise: noise,
                        sampleOptions: options.sampleOptions,
                        resilience: resilience
                    )
                    accumulateGroupedMoments(
                        group: group,
                        groupMoments: groupMoments,
                        total: &total,
                        varianceAccumulator: &varianceAccumulator
                    )
                }
            } else if let densityBackend = backend as? DensityMatrixBackend {
                method = .densityMatrix
                deviceName = MetalRuntime.deviceName
                for group in partition.groups {
                    let groupMoments = try PauliShotEstimator.estimateGroup(
                        circuit: circuit,
                        group: group,
                        engine: densityBackend.engine,
                        shots: shots,
                        rng: &rng,
                        noise: noise,
                        resilience: resilience
                    )
                    accumulateGroupedMoments(
                        group: group,
                        groupMoments: groupMoments,
                        total: &total,
                        varianceAccumulator: &varianceAccumulator
                    )
                }
            } else if let cpuDensity = backend as? CPUDensityMatrixBackend {
                method = .densityMatrix
                deviceName = "CPU"
                for group in partition.groups {
                    let groupMoments = try PauliShotEstimator.estimateGroup(
                        circuit: circuit,
                        group: group,
                        engine: cpuDensity.engine,
                        shots: shots,
                        rng: &rng,
                        noise: noise,
                        resilience: resilience
                    )
                    accumulateGroupedMoments(
                        group: group,
                        groupMoments: groupMoments,
                        total: &total,
                        varianceAccumulator: &varianceAccumulator
                    )
                }
            } else if let trajectory = backend as? TrajectoryBackend {
                method = .trajectory
                deviceName = trajectory.deviceName
                for (groupIndex, group) in partition.groups.enumerated() {
                    let groupMoments = try PauliShotEstimator.estimateGroupTrajectory(
                        circuit: circuit,
                        group: group,
                        backend: trajectory,
                        shots: shots,
                        seedBase: options.seed,
                        groupIndex: groupIndex,
                        rng: &rng,
                        noise: noise,
                        sampleOptions: options.sampleOptions,
                        resilience: resilience
                    )
                    accumulateGroupedMoments(
                        group: group,
                        groupMoments: groupMoments,
                        total: &total,
                        varianceAccumulator: &varianceAccumulator
                    )
                }
            } else if let cpuSV = backend as? CPUStatevectorBackend {
                method = .statevector
                deviceName = "CPU"
                for group in partition.groups {
                    let groupMoments = try PauliShotEstimator.estimateGroupCPU(
                        circuit: circuit,
                        group: group,
                        engine: cpuSV.engine,
                        shots: shots,
                        rng: &rng,
                        noise: noise,
                        resilience: resilience
                    )
                    accumulateGroupedMoments(
                        group: group,
                        groupMoments: groupMoments,
                        total: &total,
                        varianceAccumulator: &varianceAccumulator
                    )
                }
            } else {
                throw EstimatorError.unsupportedBackend
            }
            return ShotExpectationBox(
                value: total,
                variance: varianceAccumulator,
                method: method,
                deviceName: deviceName
            )
        }

        let suppressedShotsBody = {
            try SimulationProfiling.withGateRecordingSuppressed(shotsBody)
        }
        return try nestedUnderOuterRecorder
            ? suppressedShotsBody()
            : SimulationProfiling.timePhase("estimate", suppressedShotsBody)
    }

    /// Fold group means into the energy and add the shared-ensemble variance of `Σ cᵢ Pᵢ`.
    private func accumulateGroupedMoments(
        group: PauliCommutingGroups.Group,
        groupMoments: PauliShotEstimator.GroupShotMoments,
        total: inout QFloat,
        varianceAccumulator: inout QFloat
    ) {
        for (term, mean) in zip(group.terms, groupMoments.means) {
            total += term.coefficient * mean
        }
        varianceAccumulator += groupMoments.linearCombinationVariance
    }
}

/// Pauli-string shot estimation via basis-change + computational-basis sampling.
///
/// Prefer ``estimateGroup`` / ``estimateGroupCPU`` / ``estimateGroupTrajectory`` so
/// QWC groups share one measure circuit. Singleton wrappers remain for call sites
/// that already hold a single term.
enum PauliShotEstimator {
    /// Per-term means plus sample variance of `Σ cᵢ Pᵢ` on one shared shot ensemble.
    struct GroupShotMoments: Sendable, Equatable {
        /// ⟨Pᵢ⟩ for each term in group order.
        let means: [QFloat]
        /// Empirical `Var(Σ cᵢ Pᵢ)` over the shared histogram (covariances included).
        let linearCombinationVariance: QFloat
    }

    /// Test hook: increments once per shot ensemble (one basis-change circuit + sample).
    nonisolated(unsafe) static var samplingEnsembleCountForTests = 0

    static func resetSamplingEnsembleCountForTests() {
        samplingEnsembleCountForTests = 0
    }

    static func estimateTerm(
        circuit: QuantumCircuit,
        term: PauliTerm,
        engine: QuantumEngine,
        shots: Int,
        rng: inout QuantumRNG,
        noise: NoiseModel?,
        sampleOptions: SampleCountOptions
    ) throws -> (mean: QFloat, secondMoment: QFloat) {
        let group = singletonGroup(for: term)
        let moments = try estimateGroup(
            circuit: circuit,
            group: group,
            engine: engine,
            shots: shots,
            rng: &rng,
            noise: noise,
            sampleOptions: sampleOptions,
            resilience: .disabled
        )
        return singletonTermMoments(moments)
    }

    static func estimateTerm(
        circuit: QuantumCircuit,
        term: PauliTerm,
        engine: DensityMatrixEngine,
        shots: Int,
        rng: inout QuantumRNG,
        noise: NoiseModel?
    ) throws -> (mean: QFloat, secondMoment: QFloat) {
        let group = singletonGroup(for: term)
        let moments = try estimateGroup(
            circuit: circuit,
            group: group,
            engine: engine,
            shots: shots,
            rng: &rng,
            noise: noise,
            resilience: .disabled
        )
        return singletonTermMoments(moments)
    }

    static func estimateTerm(
        circuit: QuantumCircuit,
        term: PauliTerm,
        engine: CPUDensityMatrixEngine,
        shots: Int,
        rng: inout QuantumRNG,
        noise: NoiseModel?
    ) throws -> (mean: QFloat, secondMoment: QFloat) {
        let group = singletonGroup(for: term)
        let moments = try estimateGroup(
            circuit: circuit,
            group: group,
            engine: engine,
            shots: shots,
            rng: &rng,
            noise: noise,
            resilience: .disabled
        )
        return singletonTermMoments(moments)
    }

    static func estimateTermTrajectory(
        circuit: QuantumCircuit,
        term: PauliTerm,
        backend: TrajectoryBackend,
        shots: Int,
        seedBase: UInt64?,
        termIndex: Int = 0,
        rng: inout QuantumRNG,
        noise: NoiseModel?,
        sampleOptions: SampleCountOptions
    ) throws -> (mean: QFloat, secondMoment: QFloat) {
        let group = singletonGroup(for: term)
        let moments = try estimateGroupTrajectory(
            circuit: circuit,
            group: group,
            backend: backend,
            shots: shots,
            seedBase: seedBase,
            groupIndex: termIndex,
            rng: &rng,
            noise: noise,
            sampleOptions: sampleOptions,
            resilience: .disabled
        )
        return singletonTermMoments(moments)
    }

    static func estimateTermCPU(
        circuit: QuantumCircuit,
        term: PauliTerm,
        engine: CPUStatevectorEngine,
        shots: Int,
        rng: inout QuantumRNG,
        noise: NoiseModel?
    ) throws -> (mean: QFloat, secondMoment: QFloat) {
        let group = singletonGroup(for: term)
        let moments = try estimateGroupCPU(
            circuit: circuit,
            group: group,
            engine: engine,
            shots: shots,
            rng: &rng,
            noise: noise,
            resilience: .disabled
        )
        return singletonTermMoments(moments)
    }

    static func estimateGroup(
        circuit: QuantumCircuit,
        group: PauliCommutingGroups.Group,
        engine: QuantumEngine,
        shots: Int,
        rng: inout QuantumRNG,
        noise: NoiseModel?,
        sampleOptions: SampleCountOptions,
        resilience: ResilienceOptions = .disabled
    ) throws -> GroupShotMoments {
        guard !group.terms.isEmpty else {
            return GroupShotMoments(means: [], linearCombinationVariance: 0)
        }
        if group.measurementAxes.isEmpty {
            return constantGroupMoments(termCount: group.terms.count)
        }

        let measureCircuit = try basisChangedCircuit(
            circuit: circuit,
            measurementAxes: group.measurementAxes
        )
        samplingEnsembleCountForTests += 1
        var counts = try QuantumMeasurement.runSampleCountsRNG(
            circuit: measureCircuit,
            engine: engine,
            shots: shots,
            rng: &rng,
            noise: noise,
            options: sampleOptions
        )
        counts = try applyResilienceIfNeeded(
            counts,
            qubitCount: circuit.qubitCount,
            resilience: resilience
        )
        return groupMoments(from: counts, group: group, shots: shots)
    }

    static func estimateGroup(
        circuit: QuantumCircuit,
        group: PauliCommutingGroups.Group,
        engine: DensityMatrixEngine,
        shots: Int,
        rng: inout QuantumRNG,
        noise: NoiseModel?,
        resilience: ResilienceOptions = .disabled
    ) throws -> GroupShotMoments {
        guard !group.terms.isEmpty else {
            return GroupShotMoments(means: [], linearCombinationVariance: 0)
        }
        if group.measurementAxes.isEmpty {
            return constantGroupMoments(termCount: group.terms.count)
        }

        let measureCircuit = try basisChangedCircuit(
            circuit: circuit,
            measurementAxes: group.measurementAxes
        )
        samplingEnsembleCountForTests += 1
        var counts = try DensityMatrixShotSampler.runSampleCountsRNG(
            circuit: measureCircuit,
            engine: engine,
            shots: shots,
            rng: &rng,
            noise: noise
        )
        counts = try applyResilienceIfNeeded(
            counts,
            qubitCount: circuit.qubitCount,
            resilience: resilience
        )
        return groupMoments(from: counts, group: group, shots: shots)
    }

    static func estimateGroup(
        circuit: QuantumCircuit,
        group: PauliCommutingGroups.Group,
        engine: CPUDensityMatrixEngine,
        shots: Int,
        rng: inout QuantumRNG,
        noise: NoiseModel?,
        resilience: ResilienceOptions = .disabled
    ) throws -> GroupShotMoments {
        guard !group.terms.isEmpty else {
            return GroupShotMoments(means: [], linearCombinationVariance: 0)
        }
        if group.measurementAxes.isEmpty {
            return constantGroupMoments(termCount: group.terms.count)
        }

        let measureCircuit = try basisChangedCircuit(
            circuit: circuit,
            measurementAxes: group.measurementAxes
        )
        samplingEnsembleCountForTests += 1
        var counts = try DensityMatrixShotSampler.runSampleCountsRNG(
            circuit: measureCircuit,
            engine: engine,
            shots: shots,
            rng: &rng,
            noise: noise
        )
        counts = try applyResilienceIfNeeded(
            counts,
            qubitCount: circuit.qubitCount,
            resilience: resilience
        )
        return groupMoments(from: counts, group: group, shots: shots)
    }

    static func estimateGroupTrajectory(
        circuit: QuantumCircuit,
        group: PauliCommutingGroups.Group,
        backend: TrajectoryBackend,
        shots: Int,
        seedBase: UInt64?,
        groupIndex: Int = 0,
        rng: inout QuantumRNG,
        noise: NoiseModel?,
        sampleOptions: SampleCountOptions,
        resilience: ResilienceOptions = .disabled
    ) throws -> GroupShotMoments {
        guard !group.terms.isEmpty else {
            return GroupShotMoments(means: [], linearCombinationVariance: 0)
        }
        if group.measurementAxes.isEmpty {
            return constantGroupMoments(termCount: group.terms.count)
        }

        let measureCircuit = try basisChangedCircuit(
            circuit: circuit,
            measurementAxes: group.measurementAxes
        )
        samplingEnsembleCountForTests += 1
        // Advance per QWC group so multi-group estimates stay independent.
        let seed = (seedBase ?? UInt64(rng.nextUnitDouble() * Double(UInt64.max))) &+ UInt64(groupIndex)
        let result = try backend.run(
            circuit: measureCircuit,
            options: QuantumRunOptions(
                noise: noise,
                seed: seed,
                shots: shots,
                sampleOptions: sampleOptions
            )
        )
        var counts = result.shotCounts ?? ShotCounts(shots: shots, counts: [:])
        counts = try applyResilienceIfNeeded(
            counts,
            qubitCount: circuit.qubitCount,
            resilience: resilience
        )
        return groupMoments(from: counts, group: group, shots: shots)
    }

    static func estimateGroupCPU(
        circuit: QuantumCircuit,
        group: PauliCommutingGroups.Group,
        engine: CPUStatevectorEngine,
        shots: Int,
        rng: inout QuantumRNG,
        noise: NoiseModel?,
        resilience: ResilienceOptions = .disabled
    ) throws -> GroupShotMoments {
        guard !group.terms.isEmpty else {
            return GroupShotMoments(means: [], linearCombinationVariance: 0)
        }
        if group.measurementAxes.isEmpty {
            return constantGroupMoments(termCount: group.terms.count)
        }

        let measureCircuit = try basisChangedCircuit(
            circuit: circuit,
            measurementAxes: group.measurementAxes
        )
        samplingEnsembleCountForTests += 1
        var histogram: [Int: Int] = [:]
        for _ in 0..<shots {
            let state = try CPUStateVector(qubitCount: measureCircuit.qubitCount)
            _ = try engine.executeRNG(measureCircuit, on: state, rng: &rng, noise: noise)
            let outcome = try engine.measureCollapse(
                on: state,
                qubits: Array(0..<measureCircuit.qubitCount),
                rng: &rng,
                noise: noise
            )
            histogram[outcome, default: 0] += 1
        }
        var counts = ShotCounts(shots: shots, counts: histogram)
        counts = try applyResilienceIfNeeded(
            counts,
            qubitCount: circuit.qubitCount,
            resilience: resilience
        )
        return groupMoments(from: counts, group: group, shots: shots)
    }

    private static func singletonGroup(for term: PauliTerm) -> PauliCommutingGroups.Group {
        let axes = PauliCommutingGroups.nonIdentitySupport(term)
        return PauliCommutingGroups.Group(terms: [term], measurementAxes: axes)
    }

    private static func singletonTermMoments(
        _ moments: GroupShotMoments
    ) -> (mean: QFloat, secondMoment: QFloat) {
        // Pauli eigenvalues are ±1 ⇒ E[P²] = 1.
        (moments.means[0], 1)
    }

    private static func constantGroupMoments(termCount: Int) -> GroupShotMoments {
        GroupShotMoments(
            means: Array(repeating: 1, count: termCount),
            linearCombinationVariance: 0
        )
    }

    private static func basisChangedCircuit(
        circuit: QuantumCircuit,
        measurementAxes: [Int: Pauli]
    ) throws -> QuantumCircuit {
        var measureCircuit = try QuantumCircuit(
            qubitCount: circuit.qubitCount,
            classicalRegisters: circuit.classicalRegisters
        )
        for gate in circuit.gates {
            try measureCircuit.apply(gate)
        }

        for qubit in measurementAxes.keys.sorted() {
            switch measurementAxes[qubit] {
            case .x:
                try measureCircuit.h(qubit)
            case .y:
                try measureCircuit.sdg(qubit)
                try measureCircuit.h(qubit)
            case .z, .i, .none:
                break
            }
        }
        return measureCircuit
    }

    /// Per-term means and `Var(Σ cᵢ Pᵢ)` from one shared computational-basis histogram.
    private static func groupMoments(
        from counts: ShotCounts,
        group: PauliCommutingGroups.Group,
        shots: Int
    ) -> GroupShotMoments {
        let supports: [[Int]] = group.terms.map { term in
            term.paulis.keys.filter { term.paulis[$0] != .i }.sorted()
        }

        var meanAccum = Array(repeating: QFloat(0), count: group.terms.count)
        var sumX: QFloat = 0
        var sumX2: QFloat = 0

        for (outcome, count) in counts.counts {
            let weight = QFloat(count)
            var x: QFloat = 0
            for index in group.terms.indices {
                let eigenvalue: QFloat
                if supports[index].isEmpty {
                    eigenvalue = 1
                } else {
                    eigenvalue = parityEigenvalue(outcome: outcome, qubits: supports[index])
                }
                meanAccum[index] += eigenvalue * weight
                x += group.terms[index].coefficient * eigenvalue
            }
            sumX += x * weight
            sumX2 += x * x * weight
        }

        let invShots = QFloat(1) / QFloat(shots)
        let means = meanAccum.map { $0 * invShots }
        let meanX = sumX * invShots
        let secondMoment = sumX2 * invShots
        let variance = max(0, secondMoment - meanX * meanX)
        return GroupShotMoments(means: means, linearCombinationVariance: variance)
    }

    private static func parityEigenvalue(outcome: Int, qubits: [Int]) -> QFloat {
        var parity = 0
        for qubit in qubits {
            parity ^= (outcome >> qubit) & 1
        }
        return parity == 0 ? 1 : -1
    }
}
