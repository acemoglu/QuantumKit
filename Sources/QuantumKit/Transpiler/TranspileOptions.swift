import Foundation

/// Configuration for a named transpilation pipeline.
///
/// Existing call sites that only pass a ``BasisGateSet`` keep the prior basis-only behavior.
/// Set `couplingMap` to enable the device-aware pipeline:
/// `validate → algebraic opt? → unroll → route(+layout) → basis translate`.
public struct TranspileOptions: Sendable, Equatable {
    public var targetBasis: BasisGateSet
    public var couplingMap: CouplingMap?
    public var initialLayout: Layout?
    /// Optional gate-count budget enforced by ``PreTranspileValidationPass``.
    public var maxDepth: Int?
    /// `0` = basis translation only (default legacy behavior).
    /// `1+` = unroll (+ algebraic optimization) and, with a coupling map, route.
    public var optimizationLevel: Int

    public init(
        targetBasis: BasisGateSet = .ibmEagle,
        couplingMap: CouplingMap? = nil,
        initialLayout: Layout? = nil,
        maxDepth: Int? = nil,
        optimizationLevel: Int = 0
    ) {
        self.targetBasis = targetBasis
        self.couplingMap = couplingMap
        self.initialLayout = initialLayout
        self.maxDepth = maxDepth
        self.optimizationLevel = optimizationLevel
    }

    /// Builds the ordered pass list for these options.
    public func makePasses() throws -> [any CompilerPass] {
        var passes: [any CompilerPass] = [
            PreTranspileValidationPass(couplingMap: couplingMap, maxDepth: maxDepth)
        ]

        if optimizationLevel >= 1 {
            passes.append(AlgebraicOptimizationPass())
        }

        if let couplingMap {
            if let initialLayout {
                guard initialLayout.logicalQubitCount > 0 else {
                    throw TranspilerError.invalidLayout(reason: "initial layout is empty")
                }
            }
            passes.append(UnrollMultiQubitPass())
            passes.append(BasicSwapRoutingPass(couplingMap: couplingMap, initialLayout: initialLayout))
            passes.append(BasisTranslatorPass(targetBasis: targetBasis))
            return passes
        }

        if optimizationLevel >= 1 {
            passes.append(UnrollMultiQubitPass())
        }

        passes.append(BasisTranslatorPass(targetBasis: targetBasis))
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

        public func makePasses() -> [any CompilerPass] {
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
        try transpile(circuit, passes: preset.makePasses())
    }
}
