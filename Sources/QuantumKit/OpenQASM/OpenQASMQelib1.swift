import Foundation

/// Builtin OpenQASM 2 `qelib1.inc` catalog for QuantumKit.
///
/// OpenQASM 2 support = language core + full `qelib1.inc`; not arbitrary includes / opaque.
///
/// `include "qelib1.inc"` is accepted without reading a file from disk. The canonical
/// Qiskit `qelib1.inc` source is embedded below and registered into the importer’s
/// user-gate table on include (same expand path as user `gate` declarations).
///
/// ## Lowering policy
/// - Names in ``mappedGateNames`` lower directly to ``Gate`` (fast-paths / QuantumKit
///   extensions such as `mcx` / `mcz`). Fast-paths override embedded expansions.
/// - Other qelib1 gates expand from the embedded definitions (requires include).
/// - Capital `U` / `CX` are language primitives used inside qelib1 bodies.
/// - Non-qelib1 `include` and `opaque` remain hard errors (filesystem include off by default).
public enum OpenQASMQelib1: Sendable {
    /// Canonical include path recognized as the builtin standard library.
    public static let includeFileName = "qelib1.inc"

    /// Gate names that map directly to ``Gate`` (parameter-free and parametric).
    ///
    /// These take precedence over embedded qelib1 expansions when both exist.
    /// `ch` / `cy` lower via compact decompositions (see importer). `mcx` / `mcz`
    /// are QuantumKit extensions (not in stock qelib1). `cu1` aliases ``Gate/cp``.
    public static let mappedGateNames: Set<String> = [
        // Parameter-free
        "id", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "sxdg",
        "cx", "cz", "cy", "ch", "swap", "ccx", "cswap",
        "mcx", "mcz",
        // Parametric (numeric angle expressions)
        "u", "u1", "u2", "u3", "p",
        "rx", "ry", "rz",
        "crx", "cry", "crz", "cp", "cu1",
    ]

    /// Returns `true` when `path` refers to the builtin qelib1 include (basename match).
    public static func isBuiltinInclude(_ path: String) -> Bool {
        let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed == includeFileName { return true }
        // Allow path-style includes that end with the standard filename.
        if trimmed.hasSuffix("/" + includeFileName) { return true }
        return false
    }

    /// Canonical Qiskit `qelib1.inc` (legacy / extended) embedded as source.
    ///
    /// Parsed on `include "qelib1.inc"` so every `gate` declaration is available
    /// for expansion. Do not invent extra gate names beyond this file.
    public static let source: String = """
    // Quantum Experience (QE) Standard Header
    // file: qelib1.inc

    // --- QE Hardware primitives ---

    // 3-parameter 2-pulse single qubit gate
    gate u3(theta,phi,lambda) q { U(theta,phi,lambda) q; }
    // 2-parameter 1-pulse single qubit gate
    gate u2(phi,lambda) q { U(pi/2,phi,lambda) q; }
    // 1-parameter 0-pulse single qubit gate
    gate u1(lambda) q { U(0,0,lambda) q; }
    // controlled-NOT
    gate cx c,t { CX c,t; }
    // idle gate (identity)
    gate id a { U(0,0,0) a; }
    // idle gate (identity) with length gamma*sqglen
    gate u0(gamma) q { U(0,0,0) q; }

    // --- QE Standard Gates ---

    // generic single qubit gate
    gate u(theta,phi,lambda) q { U(theta,phi,lambda) q; }
    // phase gate
    gate p(lambda) q { U(0,0,lambda) q; }
    // Pauli gate: bit-flip
    gate x a { u3(pi,0,pi) a; }
    // Pauli gate: bit and phase flip
    gate y a { u3(pi,pi/2,pi/2) a; }
    // Pauli gate: phase flip
    gate z a { u1(pi) a; }
    // Clifford gate: Hadamard
    gate h a { u2(0,pi) a; }
    // Clifford gate: sqrt(Z) phase gate
    gate s a { u1(pi/2) a; }
    // Clifford gate: conjugate of sqrt(Z)
    gate sdg a { u1(-pi/2) a; }
    // C3 gate: sqrt(S) phase gate
    gate t a { u1(pi/4) a; }
    // C3 gate: conjugate of sqrt(S)
    gate tdg a { u1(-pi/4) a; }

    // --- Standard rotations ---
    // Rotation around X-axis
    gate rx(theta) a { u3(theta, -pi/2,pi/2) a; }
    // rotation around Y-axis
    gate ry(theta) a { u3(theta,0,0) a; }
    // rotation around Z axis
    gate rz(phi) a { u1(phi) a; }

    // --- QE Standard User-Defined Gates ---

    // sqrt(X)
    gate sx a { sdg a; h a; sdg a; }
    // inverse sqrt(X)
    gate sxdg a { s a; h a; s a; }
    // controlled-Phase
    gate cz a,b { h b; cx a,b; h b; }
    // controlled-Y
    gate cy a,b { sdg b; cx a,b; s b; }
    // swap
    gate swap a,b { cx a,b; cx b,a; cx a,b; }
    // controlled-H
    gate ch a,b {
    h b; sdg b;
    cx a,b;
    h b; t b;
    cx a,b;
    t b; h b; s b; x b; s a;
    }
    // C3 gate: Toffoli
    gate ccx a,b,c
    {
     h c;
     cx b,c; tdg c;
     cx a,c; t c;
     cx b,c; tdg c;
     cx a,c; t b; t c; h c;
     cx a,b; t a; tdg b;
     cx a,b;
    }
    // cswap (Fredkin)
    gate cswap a,b,c
    {
     cx c,b;
     ccx a,b,c;
     cx c,b;
    }
    // controlled rx rotation
    gate crx(lambda) a,b
    {
     u1(pi/2) b;
     cx a,b;
     u3(-lambda/2,0,0) b;
     cx a,b;
     u3(lambda/2,-pi/2,0) b;
    }
    // controlled ry rotation
    gate cry(lambda) a,b
    {
     ry(lambda/2) b;
     cx a,b;
     ry(-lambda/2) b;
     cx a,b;
    }
    // controlled rz rotation
    gate crz(lambda) a,b
    {
     rz(lambda/2) b;
     cx a,b;
     rz(-lambda/2) b;
     cx a,b;
    }
    // controlled phase rotation
    gate cu1(lambda) a,b
    {
     u1(lambda/2) a;
     cx a,b;
     u1(-lambda/2) b;
     cx a,b;
     u1(lambda/2) b;
    }
    gate cp(lambda) a,b
    {
     p(lambda/2) a;
     cx a,b;
     p(-lambda/2) b;
     cx a,b;
     p(lambda/2) b;
    }
    // controlled-U
    gate cu3(theta,phi,lambda) c, t
    {
      // implements controlled-U(theta,phi,lambda) with target t and control c
      u1((lambda+phi)/2) c;
      u1((lambda-phi)/2) t;
      cx c,t;
      u3(-theta/2,0,-(phi+lambda)/2) t;
      cx c,t;
      u3(theta/2,phi,0) t;
    }
    // controlled-sqrt(X)
    gate csx a,b { h b; cu1(pi/2) a,b; h b; }
    // controlled-U gate
    gate cu(theta,phi,lambda,gamma) c, t
    { p(gamma) c;
      p((lambda+phi)/2) c;
      p((lambda-phi)/2) t;
      cx c,t;
      u(-theta/2,0,-(phi+lambda)/2) t;
      cx c,t;
      u(theta/2,phi,0) t;
    }
    // two-qubit XX rotation
    gate rxx(theta) a,b
    {
      u3(pi/2, theta, 0) a;
      h b;
      cx a,b;
      u1(-theta) b;
      cx a,b;
      h b;
      u2(-pi, pi-theta) a;
    }
    // two-qubit ZZ rotation
    gate rzz(theta) a,b
    {
      cx a,b;
      u1(theta) b;
      cx a,b;
    }
    // relative-phase CCX
    gate rccx a,b,c
    {
      u2(0,pi) c;
      u1(pi/4) c;
      cx b, c;
      u1(-pi/4) c;
      cx a, c;
      u1(pi/4) c;
      cx b, c;
      u1(-pi/4) c;
      u2(0,pi) c;
    }
    // relative-phase 3-controlled X gate
    gate rc3x a,b,c,d
    {
      u2(0,pi) d;
      u1(pi/4) d;
      cx c,d;
      u1(-pi/4) d;
      u2(0,pi) d;
      cx a,d;
      u1(pi/4) d;
      cx b,d;
      u1(-pi/4) d;
      cx a,d;
      u1(pi/4) d;
      cx b,d;
      u1(-pi/4) d;
      u2(0,pi) d;
      u1(pi/4) d;
      cx c,d;
      u1(-pi/4) d;
      u2(0,pi) d;
    }
    // 3-controlled X gate
    gate c3x a,b,c,d
    {
      h d;
      p(pi/8) a;
      p(pi/8) b;
      p(pi/8) c;
      p(pi/8) d;
      cx a, b;
      p(-pi/8) b;
      cx a, b;
      cx b, c;
      p(-pi/8) c;
      cx a, c;
      p(pi/8) c;
      cx b, c;
      p(-pi/8) c;
      cx a, c;
      cx c, d;
      p(-pi/8) d;
      cx b, d;
      p(pi/8) d;
      cx c, d;
      p(-pi/8) d;
      cx a, d;
      p(pi/8) d;
      cx c, d;
      p(-pi/8) d;
      cx b, d;
      p(pi/8) d;
      cx c, d;
      p(-pi/8) d;
      cx a, d;
      h d;
    }
    // 3-controlled sqrt(X) gate, this equals the C3X gate where the CU1 rotations are -pi/8 not -pi/4
    gate c3sqrtx a,b,c,d
    {
      h d; cu1(pi/8) a,d; h d;
      cx a,b;
      h d; cu1(-pi/8) b,d; h d;
      cx a,b;
      h d; cu1(pi/8) b,d; h d;
      cx b,c;
      h d; cu1(-pi/8) c,d; h d;
      cx a,c;
      h d; cu1(pi/8) c,d; h d;
      cx b,c;
      h d; cu1(-pi/8) c,d; h d;
      cx a,c;
      h d; cu1(pi/8) c,d; h d;
    }
    // 4-controlled X gate
    gate c4x a,b,c,d,e
    {
      h e; cu1(pi/2) d,e; h e;
      c3x a,b,c,d;
      h e; cu1(-pi/2) d,e; h e;
      c3x a,b,c,d;
      c3sqrtx a,b,c,e;
    }
    """

    /// Gate names declared in ``source`` (parsed once).
    public static let embeddedGateNames: Set<String> = Set(
        embeddedGateDeclarations.map(\.name)
    )

    /// Parsed `gate` declarations from ``source`` (trusted builtin; parse failures are a programming error).
    public static let embeddedGateDeclarations: [EmbeddedGateDeclaration] = {
        do {
            var parser = try OpenQASMParser(source: source)
            let program = try parser.parse()
            var decls: [EmbeddedGateDeclaration] = []
            for statement in program.statements {
                guard case .gateDecl(let name, let params, let qubits, let body, let location) = statement else {
                    continue
                }
                decls.append(
                    EmbeddedGateDeclaration(
                        name: name,
                        params: params,
                        qubits: qubits,
                        body: body,
                        location: location
                    )
                )
            }
            return decls
        } catch {
            preconditionFailure("Failed to parse embedded qelib1.inc: \(error)")
        }
    }()

    /// A single `gate` declaration extracted from the embedded qelib1 source.
    public struct EmbeddedGateDeclaration: Equatable, Sendable {
        public let name: String
        public let params: [String]
        public let qubits: [String]
        public let body: [OpenQASMStatement]
        public let location: SourceLocation
    }
}
