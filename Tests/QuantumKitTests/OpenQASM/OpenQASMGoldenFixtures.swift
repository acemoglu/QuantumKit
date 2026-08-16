import Foundation

/// Frozen OpenQASM source strings for golden / round-trip tests.
///
/// Kept as Swift string constants (not bundle Resources) so Package.swift stays untouched.
enum OpenQASMGoldenFixtures {
    /// Bell pair in OpenQASM 2 + qelib1.
    static let bell_qasm2 = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    h q[0];
    cx q[0],q[1];
    measure q -> c;
    """

    /// Bell pair in OpenQASM 3 (`qubit` / `bit`).
    static let bell_qasm3 = """
    OPENQASM 3.0;
    qubit[2] q;
    bit[2] c;
    h q[0];
    cx q[0],q[1];
    measure q -> c;
    """

    /// Toffoli (ccx) in OpenQASM 2.
    static let toffoli_qasm2 = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    ccx q[0],q[1],q[2];
    """

    /// Teleport-ish static structure: Bell + measures + classically conditioned corrections.
    static let teleport_ish_qasm2 = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    creg c[2];
    h q[1];
    cx q[1],q[2];
    cx q[0],q[1];
    h q[0];
    measure q[0] -> c[0];
    measure q[1] -> c[1];
    if(c==1) z q[2];
    if(c==2) x q[2];
    if(c==3) z q[2];
    if(c==3) x q[2];
    """

    /// Parametric angles (`pi` expressions) in OpenQASM 2.
    static let parametric_angles_qasm2 = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    rx(pi/2) q[0];
    u3(pi/2,0,pi) q[0];
    """

    /// Bounded while via `@quantumkit.max_while_iterations` pragma (OpenQASM 3).
    static let while_bounded_qasm3 = """
    OPENQASM 3.0;
    qubit[1] q;
    bit[1] c;
    // @quantumkit.max_while_iterations 8
    while (c == 1) { x q[0]; }
    """

    /// Multi-statement braced `if` (OpenQASM 3).
    static let braced_if_qasm3 = """
    OPENQASM 3.0;
    qubit[2] q;
    bit[1] c;
    if (c == 1) {
      h q[0];
      cx q[0], q[1];
    }
    """

    /// Bounded `while` nested under braced `if` (OpenQASM 3).
    static let while_under_if_qasm3 = """
    OPENQASM 3.0;
    qubit[1] q;
    bit[1] c;
    if (c == 1) {
      // @quantumkit.max_while_iterations 5
      while (c == 1) { x q[0]; }
    }
    """

    /// qelib1 leftovers: `ch` / `cy` / `mcx` (import golden; export stays decomposed / unsupported for mcx).
    static let ch_cy_mcx_qasm2 = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[4];
    ch q[0],q[1];
    cy q[1],q[2];
    mcx q[0],q[1],q[2],q[3];
    """
}
