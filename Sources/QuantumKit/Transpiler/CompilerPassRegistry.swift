import Foundation

// MARK: - B14 lite: CompilerPass discovery (chosen extension point)
//
// Extension points already close to plugins (documented, not marketplaces):
// 1. ``CompilerPass`` + ``PassManager`` — ordered circuit transforms (THIS registry).
// 2. ``QuantumChannel`` + ``NoiseModel/adding(_:for:)`` — attach channels by target.
// 3. ``QuantumBackend`` + ``QuantumBackendFactory`` — simulation backends.
//
// Out of scope: plugin marketplace, dynamic dylib loading, sandboxing.

/// Builds a ``CompilerPass`` for in-process discovery.
///
/// Register instances on a ``CompilerPassRegistry``, then resolve by id into a
/// ``PassManager``. Existing ``PassManager(passes:)`` / ``TranspileOptions/makePasses()``
/// call sites are unchanged — the registry is opt-in.
public protocol CompilerPassFactory: Sendable {
    /// Stable lookup key (non-empty). Prefer reverse-DNS style (`"com.example.fuse"`).
    var id: String { get }

    /// Constructs a fresh pass instance for one pipeline build.
    func makePass() throws -> any CompilerPass
}

/// Errors from ``CompilerPassRegistry`` lookup / registration.
public enum CompilerPassRegistryError: Error, Equatable, Sendable {
    case emptyID
    case duplicateID(String)
    case unknownID(String)
}

/// In-process registry of named ``CompilerPassFactory`` values.
///
/// Thin discovery only: map string ids → factories → ``PassManager``. Own an instance
/// per scope (app, library, or test); QuantumKit does not ship a process-wide singleton,
/// so registration never mutates global default state.
public final class CompilerPassRegistry: @unchecked Sendable {
    /// Well-known id for an ``IdentityCompilerPass`` when clients register one.
    public static let identityPassID = "quantumkit.identity"

    private let lock = NSLock()
    private var factories: [String: any CompilerPassFactory] = [:]

    /// Empty registry — no factories until ``register`` / ``registerReplacing``.
    public init() {}

    /// `true` when no factories are registered.
    public var isEmpty: Bool {
        lock.lock()
        defer { lock.unlock() }
        return factories.isEmpty
    }

    /// Registers `factory` under ``CompilerPassFactory/id``.
    ///
    /// - Throws: ``CompilerPassRegistryError/emptyID`` or
    ///   ``CompilerPassRegistryError/duplicateID(_:)``.
    public func register(_ factory: any CompilerPassFactory) throws {
        let id = factory.id
        guard !id.isEmpty else { throw CompilerPassRegistryError.emptyID }
        lock.lock()
        defer { lock.unlock() }
        guard factories[id] == nil else {
            throw CompilerPassRegistryError.duplicateID(id)
        }
        factories[id] = factory
    }

    /// Inserts or replaces the factory for `factory.id` (empty id still rejected).
    public func registerReplacing(_ factory: any CompilerPassFactory) throws {
        let id = factory.id
        guard !id.isEmpty else { throw CompilerPassRegistryError.emptyID }
        lock.lock()
        defer { lock.unlock() }
        factories[id] = factory
    }

    public func unregister(id: String) {
        lock.lock()
        defer { lock.unlock() }
        factories.removeValue(forKey: id)
    }

    public func removeAll() {
        lock.lock()
        defer { lock.unlock() }
        factories.removeAll()
    }

    public func contains(id: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return factories[id] != nil
    }

    /// Sorted registered ids (stable for tests / diagnostics).
    public var registeredIDs: [String] {
        lock.lock()
        defer { lock.unlock() }
        return factories.keys.sorted()
    }

    public func makePass(id: String) throws -> any CompilerPass {
        let factory: (any CompilerPassFactory)?
        lock.lock()
        factory = factories[id]
        lock.unlock()
        guard let factory else {
            throw CompilerPassRegistryError.unknownID(id)
        }
        return try factory.makePass()
    }

    public func makePasses(ids: [String]) throws -> [any CompilerPass] {
        try ids.map { try makePass(id: $0) }
    }

    /// Builds a ``PassManager`` from ordered factory ids. Does not alter
    /// ``TranspileOptions`` or existing ``PassManager`` initializers.
    public func makePassManager(ids: [String]) throws -> PassManager {
        PassManager(passes: try makePasses(ids: ids))
    }
}

/// No-op ``CompilerPass``: returns the circuit unchanged.
public struct IdentityCompilerPass: CompilerPass, Sendable {
    public init() {}

    public func run(on circuit: QuantumCircuit) throws -> QuantumCircuit {
        circuit
    }
}

/// Closure-backed ``CompilerPassFactory`` for one-off / test registration.
public struct ClosureCompilerPassFactory: CompilerPassFactory {
    public let id: String
    private let builder: @Sendable () throws -> any CompilerPass

    public init(
        id: String,
        make: @escaping @Sendable () throws -> any CompilerPass
    ) {
        self.id = id
        self.builder = make
    }

    public func makePass() throws -> any CompilerPass {
        try builder()
    }
}

extension ClosureCompilerPassFactory {
    /// Factory that always yields ``IdentityCompilerPass`` under
    /// ``CompilerPassRegistry/identityPassID``.
    public static var identity: ClosureCompilerPassFactory {
        ClosureCompilerPassFactory(id: CompilerPassRegistry.identityPassID) {
            IdentityCompilerPass()
        }
    }
}
