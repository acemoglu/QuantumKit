import Foundation

/*
 PEC lite (C13) — probabilistic error cancellation for a single 1Q noise family
 =============================================================================

 Coverage
 --------
 Inverts **one** supported model as a signed quasiprobability over `{I,X,Y,Z}`:
 - Global 1Q depolarizing with engine jump probability `p`
   `Φ(ρ)=(1-p)ρ+(p/3)(XρX+YρY+ZρZ)`, or
 - General 1Q ``QuantumChannel/pauliChannel`` rates `(px,py,pz)`.

 Shot ``Estimator`` samples mitigation Paulis after each **1-qubit unitary** gate
 site, runs the weighted circuits, and returns `γ_tot · ⟨sign · E⟩`.

 Deferred (not this MVP)
 -----------------------
 Full gate-set PEC, 2Q/correlated depolarizing inverses, unitary folding, twirling,
 stacking with ZNE, noise-free recovery unitaries (recovery Paulis also see global
 dep ⇒ O(p²) residual), and Sampler quasiprobability histograms.

 Overhead
 --------
 Per site: `γ = Σ|η_σ|`. Depolarizing: `γ = (1 + 2p/3) / (1 - 4p/3)` (`p < 3/4`).
 Total for `n` independent sites: `Γ = γ^n` (shot variance scales ~ `Γ²`).
 */

/// Opt-in PEC lite knobs (C13). Presence on ``ResilienceOptions/pec`` enables mitigation.
public struct PECOptions: Sendable, Equatable {
    /// Channel whose inverse QPR is sampled at each 1Q gate site.
    public var channel: PECInverseChannel
    /// Number of distinct mitigation circuits. `nil` → use the Estimator shot count.
    public var circuitSamples: Int?

    public init(
        channel: PECInverseChannel = .globalDepolarizing,
        circuitSamples: Int? = nil
    ) {
        self.channel = channel
        self.circuitSamples = circuitSamples
    }

    public static let `default` = PECOptions()

    public var isActive: Bool { true }
}

/// Which inverse quasiprobability to use for PEC lite.
public enum PECInverseChannel: Sendable, Equatable {
    /// Invert ``NoiseModel/depolarizingProbability`` (1Q jump form).
    case globalDepolarizing
    /// Invert a Pauli channel `(1-px-py-pz)ρ + px XρX + py YρY + pz ZρZ`.
    /// Physical noise must still be global depolarizing with `p = px+py+pz` and
    /// `px == py == pz` (within tol); unequal rates need localized DM noise (deferred).
    case pauliChannel(px: QFloat, py: QFloat, pz: QFloat)
}

/// Signed quasiprobability over single-qubit Paulis for one noise site.
public struct PauliQuasiprobability: Sendable, Equatable {
    public let etaI: QFloat
    public let etaX: QFloat
    public let etaY: QFloat
    public let etaZ: QFloat

    public init(etaI: QFloat, etaX: QFloat, etaY: QFloat, etaZ: QFloat) {
        self.etaI = etaI
        self.etaX = etaX
        self.etaY = etaY
        self.etaZ = etaZ
    }

    /// Sampling overhead `γ = Σ|η|` for one site.
    public var gamma: QFloat {
        abs(etaI) + abs(etaX) + abs(etaY) + abs(etaZ)
    }

    /// Sample `σ` with probability `|η_σ|/γ`; returns Pauli (nil = I) and `sign(η_σ)`.
    public func sample(rng: inout QuantumRNG) -> (pauli: Pauli, sign: QFloat) {
        let g = gamma
        guard g > 0 else { return (.i, 1) }
        let r = Double(rng.nextUnitFloat()) * Double(g)
        var cdf: Double = 0
        let entries: [(Pauli, QFloat)] = [
            (.i, etaI), (.x, etaX), (.y, etaY), (.z, etaZ),
        ]
        for (pauli, eta) in entries {
            cdf += Double(abs(eta))
            if r <= cdf {
                let sign: QFloat = eta >= 0 ? 1 : -1
                return (pauli, sign)
            }
        }
        let last = entries[3]
        return (last.0, last.1 >= 0 ? 1 : -1)
    }
}

/// Metadata attached to ``EstimatorResult`` when PEC lite ran.
public struct PECMetadata: Sendable, Equatable {
    public let channel: PECInverseChannel
    /// Per-site overhead `γ`.
    public let gammaPerSite: QFloat
    /// Total overhead `Γ = γ^n` for `n` PEC sites.
    public let gammaTotal: QFloat
    public let siteCount: Int
    public let circuitSamples: Int
    /// Approximate shot multiplier `Γ²` (variance inflation vs an ideal inverse).
    public let shotMultiplier: QFloat

    public init(
        channel: PECInverseChannel,
        gammaPerSite: QFloat,
        gammaTotal: QFloat,
        siteCount: Int,
        circuitSamples: Int
    ) {
        self.channel = channel
        self.gammaPerSite = gammaPerSite
        self.gammaTotal = gammaTotal
        self.siteCount = siteCount
        self.circuitSamples = circuitSamples
        self.shotMultiplier = gammaTotal * gammaTotal
    }
}

public enum PECError: Error, Equatable {
    case incompatibleWithZNE
    case missingNoiseModel
    case unsupportedNoiseModel(String)
    case nonInvertibleChannel(String)
    case unsupportedMultiQubitGate(String)
    case emptyCircuitNoPECSites
    case invalidCircuitSampleCount(Int)
    /// Explicit ``PECOptions/circuitSamples`` must not exceed the Estimator shot budget.
    case circuitSamplesExceedShots(samples: Int, shots: Int)
    case pauliChannelMismatch(expectedP: QFloat, actualDepolarizing: QFloat)
}

/// Host-side PEC lite helpers (C13).
public enum ProbabilisticErrorCancellation {

    /// QPR for engine 1Q depolarizing jump probability `p` (`p < 3/4`).
    public static func depolarizingQuasiprobability(probability p: QFloat) throws -> PauliQuasiprobability {
        guard p >= 0, p < 0.75 - 1e-6 else {
            throw PECError.nonInvertibleChannel(
                "depolarizing p=\(p) requires 0 ≤ p < 3/4 for a positive Pauli eigenvalue λ=1-4p/3"
            )
        }
        let lambda = 1 - (4 * p) / 3
        guard abs(lambda) > 1e-8 else {
            throw PECError.nonInvertibleChannel("depolarizing eigenvalue λ≈0")
        }
        let etaI = (lambda + 3) / (4 * lambda)
        let etaP = (lambda - 1) / (4 * lambda)
        return PauliQuasiprobability(etaI: etaI, etaX: etaP, etaY: etaP, etaZ: etaP)
    }

    /// QPR for ``QuantumChannel/pauliChannel`` rates (`pI = 1-px-py-pz`).
    public static func pauliChannelQuasiprobability(
        px: QFloat,
        py: QFloat,
        pz: QFloat
    ) throws -> PauliQuasiprobability {
        let cx = min(max(px, 0), 1)
        let cy = min(max(py, 0), 1)
        let cz = min(max(pz, 0), 1)
        let sum = cx + cy + cz
        guard sum <= 1 + 1e-6 else {
            throw PECError.nonInvertibleChannel("pauliChannel probabilities sum to \(sum) > 1")
        }
        let pI = max(0, 1 - sum)
        let lambdaI: QFloat = 1
        let lambdaX = pI + cx - cy - cz
        let lambdaY = pI - cx + cy - cz
        let lambdaZ = pI - cx - cy + cz
        for (name, lambda) in [("X", lambdaX), ("Y", lambdaY), ("Z", lambdaZ), ("I", lambdaI)] {
            guard abs(lambda) > 1e-8 else {
                throw PECError.nonInvertibleChannel("pauliChannel eigenvalue λ_\(name)≈0")
            }
        }
        let iL = 1 / lambdaI
        let xL = 1 / lambdaX
        let yL = 1 / lambdaY
        let zL = 1 / lambdaZ
        return PauliQuasiprobability(
            etaI: 0.25 * (iL + xL + yL + zL),
            etaX: 0.25 * (iL + xL - yL - zL),
            etaY: 0.25 * (iL - xL + yL - zL),
            etaZ: 0.25 * (iL - xL - yL + zL)
        )
    }

    /// Resolve QPR from options + physical ``NoiseModel``.
    public static func quasiprobability(
        channel: PECInverseChannel,
        noise: NoiseModel
    ) throws -> PauliQuasiprobability {
        switch channel {
        case .globalDepolarizing:
            return try depolarizingQuasiprobability(probability: noise.depolarizingProbability)
        case .pauliChannel(let px, let py, let pz):
            let qpr = try pauliChannelQuasiprobability(px: px, py: py, pz: pz)
            // Physical engines expose unequal Pauli noise only via localized DM rules (deferred).
            let tol: QFloat = 1e-5
            guard abs(px - py) <= tol, abs(py - pz) <= tol else {
                throw PECError.unsupportedNoiseModel(
                    "unequal pauliChannel PEC requires localized DM noise (deferred); use equal px=py=pz or globalDepolarizing"
                )
            }
            let p = px + py + pz
            guard abs(noise.depolarizingProbability - p) <= tol else {
                throw PECError.pauliChannelMismatch(
                    expectedP: p,
                    actualDepolarizing: noise.depolarizingProbability
                )
            }
            return qpr
        }
    }

    /// Validate noise + circuit for PEC lite; returns 1Q unitary gate indices that are sites.
    public static func validatedPECSites(
        circuit: QuantumCircuit,
        noise: NoiseModel?
    ) throws -> [Int] {
        guard let noise else { throw PECError.missingNoiseModel }
        try validateNoiseModelForPEC(noise)

        var sites: [Int] = []
        for (index, gate) in circuit.gates.enumerated() {
            if isNonUnitaryOrIdle(gate) { continue }
            let qubits = gate.affectedQubits
            if qubits.count >= 2 {
                throw PECError.unsupportedMultiQubitGate(String(describing: gate))
            }
            if qubits.count == 1 {
                sites.append(index)
            }
        }
        guard !sites.isEmpty else { throw PECError.emptyCircuitNoPECSites }
        return sites
    }

    public static func validateNoiseModelForPEC(_ noise: NoiseModel) throws {
        guard noise.appliesDepolarizing else {
            throw PECError.unsupportedNoiseModel("PEC lite requires global depolarizingProbability > 0")
        }
        if noise.appliesAmplitudeDamping {
            throw PECError.unsupportedNoiseModel("amplitude damping not supported by PEC lite")
        }
        if noise.appliesPhaseDamping {
            throw PECError.unsupportedNoiseModel("phase damping not supported by PEC lite")
        }
        if noise.hasLocalizedGateNoise {
            throw PECError.unsupportedNoiseModel("localized gate noise not supported by PEC lite")
        }
        if noise.hasPreparationNoise {
            throw PECError.unsupportedNoiseModel("preparation/reset noise not supported by PEC lite")
        }
        if noise.hasIdleNoise {
            throw PECError.unsupportedNoiseModel("idle thermal noise not supported by PEC lite")
        }
        if noise.measurementDephasingProbability > 0 {
            throw PECError.unsupportedNoiseModel("measurement dephasing not supported by PEC lite")
        }
        // Readout flips / confusion are not part of the inverse QPR — refuse rather than
        // silently return a biased mitigated expectation. Use ``ResilienceOptions/readoutMitigation``
        // for assignment-matrix correction (orthogonal to PEC lite).
        if noise.appliesReadoutError {
            throw PECError.unsupportedNoiseModel(
                "NoiseModel readout error/confusion not supported by PEC lite; use ResilienceOptions.readoutMitigation"
            )
        }
    }

    /// Insert sampled Paulis after each PEC site (identity omitted).
    public static func circuitByInsertingMitigationPaulis(
        _ circuit: QuantumCircuit,
        siteGateIndices: [Int],
        paulish: [Pauli]
    ) throws -> QuantumCircuit {
        precondition(siteGateIndices.count == paulish.count)
        var siteSet = Set(siteGateIndices)
        var pauliBySite: [Int: Pauli] = [:]
        for (site, pauli) in zip(siteGateIndices, paulish) {
            pauliBySite[site] = pauli
        }

        var out = try QuantumCircuit(
            qubitCount: circuit.qubitCount,
            classicalRegisters: circuit.classicalRegisters
        )
        for (index, gate) in circuit.gates.enumerated() {
            try out.apply(gate)
            guard siteSet.contains(index), let pauli = pauliBySite[index], pauli != .i else {
                continue
            }
            let qubit = gate.affectedQubits[0]
            switch pauli {
            case .x: try out.x(qubit)
            case .y: try out.y(qubit)
            case .z: try out.z(qubit)
            case .i: break
            }
        }
        return out
    }

    public static func fingerprintToken(for options: PECOptions) -> String {
        switch options.channel {
        case .globalDepolarizing:
            return "pec:globalDepolarizing:samples:\(options.circuitSamples.map(String.init) ?? "auto")"
        case .pauliChannel(let px, let py, let pz):
            return "pec:pauliChannel:\(px),\(py),\(pz):samples:\(options.circuitSamples.map(String.init) ?? "auto")"
        }
    }

    private static func isNonUnitaryOrIdle(_ gate: Gate) -> Bool {
        switch gate {
        case .id, .barrier, .delay, .measure, .reset, .initialize, .c_if:
            return true
        default:
            return false
        }
    }
}
