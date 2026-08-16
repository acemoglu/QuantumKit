import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - Full qelib1 import coverage
    //
    // OpenQASM 2 support = language core + full qelib1.inc; not arbitrary includes / opaque.

    /// Snapshot: gates that historically needed expand (or Fast-path) must import cleanly.
    func testOpenQASM2Qelib1CoverageSnapshotGates() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[5];
        cu3(pi/2,0,pi/4) q[0],q[1];
        csx q[0],q[1];
        rxx(pi/8) q[0],q[1];
        rzz(0.25) q[1],q[2];
        rccx q[0],q[1],q[2];
        c3x q[0],q[1],q[2],q[3];
        c4x q[0],q[1],q[2],q[3],q[4];
        cu(pi/2,0,0,pi/4) q[0],q[1];
        cp(pi/2) q[0],q[1];
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        XCTAssertFalse(circuit.gates.isEmpty)
        XCTAssertEqual(circuit.qubitCount, 5)

        // Fast-path gates stay atomic.
        XCTAssertTrue(circuit.gates.contains { gate in
            if case .rxx = gate { return true }
            return false
        })
        XCTAssertTrue(circuit.gates.contains { gate in
            if case .rzz = gate { return true }
            return false
        })
        XCTAssertTrue(circuit.gates.contains { gate in
            if case .cp = gate { return true }
            return false
        })
    }

    /// Every embedded qelib1 gate name is callable with a minimal arity-correct program.
    func testOpenQASM2Qelib1EveryEmbeddedGateImports() throws {
        let samples: [(String, String)] = [
            ("u3", "u3(pi/2,0,0) q[0];"),
            ("u2", "u2(0,pi) q[0];"),
            ("u1", "u1(pi/4) q[0];"),
            ("cx", "cx q[0],q[1];"),
            ("id", "id q[0];"),
            ("u0", "u0(1) q[0];"),
            ("u", "u(pi/2,0,0) q[0];"),
            ("p", "p(pi/2) q[0];"),
            ("x", "x q[0];"),
            ("y", "y q[0];"),
            ("z", "z q[0];"),
            ("h", "h q[0];"),
            ("s", "s q[0];"),
            ("sdg", "sdg q[0];"),
            ("t", "t q[0];"),
            ("tdg", "tdg q[0];"),
            ("rx", "rx(pi/3) q[0];"),
            ("ry", "ry(pi/3) q[0];"),
            ("rz", "rz(pi/3) q[0];"),
            ("sx", "sx q[0];"),
            ("sxdg", "sxdg q[0];"),
            ("cz", "cz q[0],q[1];"),
            ("cy", "cy q[0],q[1];"),
            ("swap", "swap q[0],q[1];"),
            ("ch", "ch q[0],q[1];"),
            ("ccx", "ccx q[0],q[1],q[2];"),
            ("cswap", "cswap q[0],q[1],q[2];"),
            ("crx", "crx(pi/5) q[0],q[1];"),
            ("cry", "cry(pi/5) q[0],q[1];"),
            ("crz", "crz(pi/5) q[0],q[1];"),
            ("cu1", "cu1(pi/6) q[0],q[1];"),
            ("cp", "cp(pi/6) q[0],q[1];"),
            ("cu3", "cu3(pi/2,0,0) q[0],q[1];"),
            ("csx", "csx q[0],q[1];"),
            ("cu", "cu(pi/2,0,0,0) q[0],q[1];"),
            ("rxx", "rxx(pi/7) q[0],q[1];"),
            ("rzz", "rzz(pi/7) q[0],q[1];"),
            ("rccx", "rccx q[0],q[1],q[2];"),
            ("rc3x", "rc3x q[0],q[1],q[2],q[3];"),
            ("c3x", "c3x q[0],q[1],q[2],q[3];"),
            ("c3sqrtx", "c3sqrtx q[0],q[1],q[2],q[3];"),
            ("c4x", "c4x q[0],q[1],q[2],q[3],q[4];"),
        ]

        let embedded = OpenQASMQelib1.embeddedGateNames
        XCTAssertEqual(Set(samples.map(\.0)), embedded, "Coverage list must match embedded qelib1 gates")

        for (name, call) in samples {
            let source = """
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[5];
            \(call)
            """
            do {
                let circuit = try OpenQASM2Importer().`import`(source: source)
                XCTAssertFalse(
                    circuit.gates.isEmpty,
                    "Gate '\(name)' produced an empty circuit"
                )
            } catch {
                XCTFail("Gate '\(name)' failed to import: \(error)")
            }
        }
    }

    func testOpenQASM2Qelib1WithoutIncludeExpandOnlyGatesFail() {
        // Expand-only gates require include so they are registered.
        let source = """
        OPENQASM 2.0;
        qreg q[2];
        cu3(pi/2,0,0) q[0],q[1];
        """
        XCTAssertThrowsError(try OpenQASM2Importer().`import`(source: source)) { error in
            guard case OpenQASMError.semanticError(_, _, let message) = error else {
                return XCTFail("Expected semanticError, got \(error)")
            }
            XCTAssertTrue(message.contains("Unknown or unmapped"), message)
        }
    }

    func testOpenQASM2ForeignIncludeStillHardError() {
        let source = """
        OPENQASM 2.0;
        include "other.inc";
        qreg q[1];
        """
        XCTAssertThrowsError(try OpenQASM2Importer().`import`(source: source)) { error in
            guard case OpenQASMError.unsupported(_, _, let feature, let message) = error else {
                return XCTFail("Expected unsupported, got \(error)")
            }
            XCTAssertEqual(feature, "include")
            XCTAssertTrue(message.contains("qelib1.inc"), message)
        }
    }

    func testOpenQASM2OpaqueStillHardError() {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        opaque foo q;
        """
        XCTAssertThrowsError(try OpenQASM2Importer().`import`(source: source)) { error in
            guard case OpenQASMError.unsupported(_, _, let feature, _) = error else {
                return XCTFail("Expected unsupported, got \(error)")
            }
            XCTAssertEqual(feature, "opaque")
        }
    }

    func testOpenQASM2Qelib1CPRoundTrip() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        cp(pi/2) q[0],q[1];
        """
        let first = try OpenQASM2Importer().`import`(source: source)
        let exported = try OpenQASM2Exporter().export(first)
        XCTAssertTrue(exported.contains("cp("), exported)
        let second = try OpenQASM2Importer().`import`(source: exported)
        XCTAssertEqual(second.gates, first.gates)
    }

    func testOpenQASM2Qelib1CSXAndRCCXImport() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        csx q[0],q[1];
        rccx q[0],q[1],q[2];
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        // Both expand into mapped primitives (h/cu1/… and u2/u1/cx/…).
        XCTAssertGreaterThan(circuit.gates.count, 2)
    }
}
