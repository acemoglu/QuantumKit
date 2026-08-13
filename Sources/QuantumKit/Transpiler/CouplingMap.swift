import Foundation

/// Undirected connectivity graph for a device's physical qubits.
///
/// Two-qubit gates may only execute on edges of this map. Routing inserts ``Gate/swap``
/// operations to satisfy that constraint without changing the logical unitary (up to the
/// inserted SWAPs, which are tracked by the evolving layout).
public struct CouplingMap: Sendable, Equatable, Codable {
    public let qubitCount: Int
    /// Canonical undirected edges with `first < second`.
    public let edges: [(Int, Int)]

    private let adjacency: [Set<Int>]

    public static func == (lhs: CouplingMap, rhs: CouplingMap) -> Bool {
        lhs.qubitCount == rhs.qubitCount
            && lhs.edges.count == rhs.edges.count
            && zip(lhs.edges, rhs.edges).allSatisfy { $0.0 == $1.0 && $0.1 == $1.1 }
    }

    public init(qubitCount: Int, edges: [(Int, Int)]) throws {
        guard qubitCount > 0 else {
            throw TranspilerError.invalidCouplingMap(reason: "qubitCount must be positive")
        }

        var adjacency = Array(repeating: Set<Int>(), count: qubitCount)
        var canonical: [(Int, Int)] = []
        var seen = Set<String>()

        for edge in edges {
            let a = edge.0
            let b = edge.1
            guard a >= 0, a < qubitCount, b >= 0, b < qubitCount else {
                throw TranspilerError.invalidCouplingMap(
                    reason: "edge (\(a), \(b)) is outside qubit range 0..<\(qubitCount)"
                )
            }
            guard a != b else {
                throw TranspilerError.invalidCouplingMap(reason: "self-loop (\(a), \(b)) is not allowed")
            }

            let lo = min(a, b)
            let hi = max(a, b)
            let key = "\(lo)-\(hi)"
            guard seen.insert(key).inserted else { continue }

            canonical.append((lo, hi))
            adjacency[lo].insert(hi)
            adjacency[hi].insert(lo)
        }

        self.qubitCount = qubitCount
        self.edges = canonical.sorted { lhs, rhs in
            if lhs.0 != rhs.0 { return lhs.0 < rhs.0 }
            return lhs.1 < rhs.1
        }
        self.adjacency = adjacency
    }

    /// Path graph: 0—1—…—(n-1).
    public static func linear(_ qubitCount: Int) throws -> CouplingMap {
        let edges = (0..<(qubitCount - 1)).map { ($0, $0 + 1) }
        return try CouplingMap(qubitCount: qubitCount, edges: edges)
    }

    /// Complete graph: every pair is adjacent.
    public static func fullyConnected(_ qubitCount: Int) throws -> CouplingMap {
        var edges: [(Int, Int)] = []
        for i in 0..<qubitCount {
            for j in (i + 1)..<qubitCount {
                edges.append((i, j))
            }
        }
        return try CouplingMap(qubitCount: qubitCount, edges: edges)
    }

    public func areAdjacent(_ a: Int, _ b: Int) -> Bool {
        guard a >= 0, a < qubitCount, b >= 0, b < qubitCount else { return false }
        return adjacency[a].contains(b)
    }

    public func neighbors(of qubit: Int) -> Set<Int> {
        guard qubit >= 0, qubit < qubitCount else { return [] }
        return adjacency[qubit]
    }

    /// Shortest path from `source` to `target` inclusive, or `nil` if disconnected.
    public func shortestPath(from source: Int, to target: Int) -> [Int]? {
        guard source >= 0, source < qubitCount, target >= 0, target < qubitCount else {
            return nil
        }
        if source == target { return [source] }

        var parent = Array(repeating: -1, count: qubitCount)
        var visited = Array(repeating: false, count: qubitCount)
        var queue: [Int] = [source]
        visited[source] = true
        var head = 0

        while head < queue.count {
            let node = queue[head]
            head += 1
            if node == target { break }
            for next in adjacency[node].sorted() {
                guard !visited[next] else { continue }
                visited[next] = true
                parent[next] = node
                queue.append(next)
            }
        }

        guard visited[target] else { return nil }

        var path = [target]
        var current = target
        while current != source {
            current = parent[current]
            path.append(current)
        }
        return path.reversed()
    }
}

extension CouplingMap {
    private enum CodingKeys: String, CodingKey {
        case qubitCount
        case edges
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let qubitCount = try container.decode(Int.self, forKey: .qubitCount)
        let rawEdges = try container.decode([[Int]].self, forKey: .edges)
        let edges = try rawEdges.map { pair -> (Int, Int) in
            guard pair.count == 2 else {
                throw TranspilerError.invalidCouplingMap(reason: "each edge must have exactly two endpoints")
            }
            return (pair[0], pair[1])
        }
        try self.init(qubitCount: qubitCount, edges: edges)
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(qubitCount, forKey: .qubitCount)
        try container.encode(edges.map { [$0.0, $0.1] }, forKey: .edges)
    }
}
