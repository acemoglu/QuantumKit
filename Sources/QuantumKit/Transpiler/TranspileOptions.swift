import Foundation

/// Configuration for a named transpilation pipeline.
///
/// Existing call sites that only pass a ``BasisGateSet`` keep the prior basis-only behavior.
/// Set `couplingMap` to enable the device-aware pipeline:
/// `validate → algebraic opt? → clifford/local? → unroll → route(+layout) → basis → schedule?`.
///
/// ## Optimization levels
/// - `0`: validate + basis translate (legacy). With a coupling map, still routes (unroll → route → basis).
/// - `1`: algebraic optimization + unroll (and route when coupled). Typically fewer gates than level 0
///   on circuits with cancellable identities.
/// - `2+`: level 1 plus ``CliffordSimplificationPass`` and ``LocalUnitarySynthesisPass`` (before and
///   after basis translation). Measurably fewer gates/depth on Clifford / adjacent-rotation fixtures.
public struct TranspileOptions: Sendable, Equatable {
    public var targetBasis: BasisGateSet
    public var couplingMap: CouplingMap?
    public var initialLayout: Layout?
    /// Optional gate-count budget enforced by ``PreTranspileValidationPass``.
    public var maxDepth: Int?
    /// `0` = basis translation only (default legacy behavior).
    /// `1+` = unroll (+ algebraic optimization) and, with a coupling map, route.
    /// `2+` = Clifford + local unitary synthesis.
    public var optimizationLevel: Int
    /// Seeds stochastic layout / routing choices. Same seed → identical transpiled circuit.
    public var seedTranspiler: UInt64?
    /// Multi-controlled synthesis strategy (default preserves ancilla-free Barenco path).
    public var controlledSynthesis: ControlledGateSynthesisStrategy
    /// Opt-in compiler ancilla allocation for strategies that need scratch qubits.
    public var enableAncillaAllocation: Bool
    /// When ancilla allocation is on, disable reuse (allocate-every-time width upper bound).
    public var disableAncillaReuse: Bool
    /// Optional non-pulse ASAP/ALAP scheduling after basis translation.
    public var scheduling: SchedulingMethod?
    public var gateDurations: GateDurationTable
    /// Retain instruction metadata through identity-preserving remaps / non-expanding keep-paths.
    /// Expanding passes (unroll / basis expansion / schedule delay insertion) still strip metadata
    /// because instruction indices no longer align 1:1. Default is `false` (strip everywhere).
    public var preserveInstructionMetadata: Bool

    public init(
        targetBasis: BasisGateSet = .ibmEagle,
        couplingMap: CouplingMap? = nil,
        initialLayout: Layout? = nil,
        maxDepth: Int? = nil,
        optimizationLevel: Int = 0,
        seedTranspiler: UInt64? = nil,
        controlledSynthesis: ControlledGateSynthesisStrategy = .ancillaFree,
        enableAncillaAllocation: Bool = false,
        disableAncillaReuse: Bool = false,
        scheduling: SchedulingMethod? = nil,
        gateDurations: GateDurationTable = .uniform(1),
        preserveInstructionMetadata: Bool = false
    ) {
        self.targetBasis = targetBasis
        self.couplingMap = couplingMap
        self.initialLayout = initialLayout
        self.maxDepth = maxDepth
        self.optimizationLevel = optimizationLevel
        self.seedTranspiler = seedTranspiler
        self.controlledSynthesis = controlledSynthesis
        self.enableAncillaAllocation = enableAncillaAllocation
        self.disableAncillaReuse = disableAncillaReuse
        self.scheduling = scheduling
        self.gateDurations = gateDurations
        self.preserveInstructionMetadata = preserveInstructionMetadata
    }

    /// Builds the ordered pass list for these options.
    public func makePasses() throws -> [any CompilerPass] {
        if controlledSynthesis == .vChainAncilla && !enableAncillaAllocation {
            throw AncillaAllocationError.ancillaAllocationDisabled(strategy: .vChainAncilla)
        }

        var passes: [any CompilerPass] = [
            PreTranspileValidationPass(couplingMap: couplingMap, maxDepth: maxDepth)
        ]

        if optimizationLevel >= 1 {
            passes.append(AlgebraicOptimizationPass())
        }
        if optimizationLevel >= 2 {
            passes.append(CliffordSimplificationPass())
            passes.append(LocalUnitarySynthesisPass())
        }

        let unroll = UnrollMultiQubitPass(
            controlledSynthesis: controlledSynthesis,
            enableAncillaAllocation: enableAncillaAllocation,
            disableAncillaReuse: disableAncillaReuse
        )

        if let couplingMap {
            if let initialLayout {
                guard initialLayout.logicalQubitCount > 0 else {
                    throw TranspilerError.invalidLayout(reason: "initial layout is empty")
                }
            }
            passes.append(unroll)
            passes.append(
                BasicSwapRoutingPass(
                    couplingMap: couplingMap,
                    initialLayout: initialLayout,
                    seed: seedTranspiler,
                    preserveInstructionMetadata: preserveInstructionMetadata
                )
            )
            passes.append(
                BasisTranslatorPass(
                    targetBasis: targetBasis,
                    preserveInstructionMetadata: preserveInstructionMetadata
                )
            )
            if optimizationLevel >= 2 {
                passes.append(LocalUnitarySynthesisPass())
            }
            if let scheduling {
                passes.append(SchedulingPass(durations: gateDurations, method: scheduling))
            }
            return passes
        }

        if optimizationLevel >= 1 || controlledSynthesis != .ancillaFree || enableAncillaAllocation {
            passes.append(unroll)
        }

        passes.append(
            BasisTranslatorPass(
                targetBasis: targetBasis,
                preserveInstructionMetadata: preserveInstructionMetadata
            )
        )
        if optimizationLevel >= 2 {
            passes.append(LocalUnitarySynthesisPass())
        }
        if let scheduling {
            passes.append(SchedulingPass(durations: gateDurations, method: scheduling))
        }
        return passes
    }
}

extension Transpiler {
    /// Named factory for the default pass manager pipelines.
    public enum Preset: Sendable, Equatable {
        /// Basis translation only (legacy default).
        case basisOnly(BasisGateSet)
        /// Validate, algebraically optimize, unroll, route, basis-translate.
        case deviceAware(couplingMap: CouplingMap, basis: BasisGateSet, layout: Layout?)
        /// Device-aware with explicit seed and optimization level.
        case deviceAwareSeeded(
            couplingMap: CouplingMap,
            basis: BasisGateSet,
            layout: Layout?,
            seed: UInt64?,
            optimizationLevel: Int
        )

        public func makePasses() throws -> [any CompilerPass] {
            switch self {
            case .basisOnly(let basis):
                return [BasisTranslatorPass(targetBasis: basis)]
            case .deviceAware(let couplingMap, let basis, let layout):
                return [
                    PreTranspileValidationPass(couplingMap: couplingMap),
                    AlgebraicOptimizationPass(),
                    UnrollMultiQubitPass(),
                    BasicSwapRoutingPass(couplingMap: couplingMap, initialLayout: layout),
                    BasisTranslatorPass(targetBasis: basis),
                ]
            case .deviceAwareSeeded(let couplingMap, let basis, let layout, let seed, let level):
                let options = TranspileOptions(
                    targetBasis: basis,
                    couplingMap: couplingMap,
                    initialLayout: layout,
                    optimizationLevel: level,
                    seedTranspiler: seed
                )
                return try options.makePasses()
            }
        }
    }

    /// Device-aware / configurable transpile entry point. Additive — does not change
    /// ``transpile(_:targetBasis:)``.
    public static func transpile(
        _ circuit: QuantumCircuit,
        options: TranspileOptions
    ) throws -> QuantumCircuit {
        try transpile(circuit, passes: try options.makePasses())
    }

    /// Runs a named ``Preset`` pipeline.
    public static func transpile(
        _ circuit: QuantumCircuit,
        preset: Preset
    ) throws -> QuantumCircuit {
        try transpile(circuit, passes: try preset.makePasses())
    }
}
