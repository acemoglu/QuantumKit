extension QuantumCircuit {

    @discardableResult
    public mutating func h(_ target: Int) throws -> QuantumCircuit {
        try applyValidated(.h(target: target))
        return self
    }

    @discardableResult
    public mutating func cx(_ control: Int, _ target: Int) throws -> QuantumCircuit {
        try applyValidated(.cx(control: control, target: target))
        return self
    }

    @discardableResult
    public mutating func ccx(_ control1: Int, _ control2: Int, _ target: Int) throws -> QuantumCircuit {
        try applyValidated(.ccx(control1: control1, control2: control2, target: target))
        return self
    }

    @discardableResult
    public mutating func rx(theta: QFloatExpr, _ target: Int) throws -> QuantumCircuit {
        try applyValidated(.rx(theta: theta, target: target))
        return self
    }

    @discardableResult
    public mutating func rx(theta: QFloat, _ target: Int) throws -> QuantumCircuit {
        try rx(theta: QFloatExpr(theta), target)
    }

    @discardableResult
    public mutating func rz(theta: QFloatExpr, _ target: Int) throws -> QuantumCircuit {
        try applyValidated(.rz(theta: theta, target: target))
        return self
    }

    @discardableResult
    public mutating func rz(theta: QFloat, _ target: Int) throws -> QuantumCircuit {
        try rz(theta: QFloatExpr(theta), target)
    }

    @discardableResult
    public mutating func ry(theta: QFloatExpr, _ target: Int) throws -> QuantumCircuit {
        try applyValidated(.ry(theta: theta, target: target))
        return self
    }

    @discardableResult
    public mutating func ry(theta: QFloat, _ target: Int) throws -> QuantumCircuit {
        try ry(theta: QFloatExpr(theta), target)
    }

    @discardableResult
    public mutating func s(_ target: Int) throws -> QuantumCircuit {
        try applyValidated(.s(target: target))
        return self
    }

    @discardableResult
    public mutating func t(_ target: Int) throws -> QuantumCircuit {
        try applyValidated(.t(target: target))
        return self
    }

    @discardableResult
    public mutating func sdg(_ target: Int) throws -> QuantumCircuit {
        try applyValidated(.sdg(target: target))
        return self
    }

    @discardableResult
    public mutating func tdg(_ target: Int) throws -> QuantumCircuit {
        try applyValidated(.tdg(target: target))
        return self
    }

    @discardableResult
    public mutating func sx(_ target: Int) throws -> QuantumCircuit {
        try applyValidated(.sx(target: target))
        return self
    }

    @discardableResult
    public mutating func sxdg(_ target: Int) throws -> QuantumCircuit {
        try applyValidated(.sxdg(target: target))
        return self
    }

    @discardableResult
    public mutating func p(theta: QFloatExpr, _ target: Int) throws -> QuantumCircuit {
        try applyValidated(.p(theta: theta, target: target))
        return self
    }

    @discardableResult
    public mutating func p(theta: QFloat, _ target: Int) throws -> QuantumCircuit {
        try p(theta: QFloatExpr(theta), target)
    }

    @discardableResult
    public mutating func u(
        theta: QFloatExpr,
        phi: QFloatExpr,
        lambda: QFloatExpr,
        _ target: Int
    ) throws -> QuantumCircuit {
        try applyValidated(.u(theta: theta, phi: phi, lambda: lambda, target: target))
        return self
    }

    @discardableResult
    public mutating func u(theta: QFloat, phi: QFloat, lambda: QFloat, _ target: Int) throws -> QuantumCircuit {
        try u(theta: QFloatExpr(theta), phi: QFloatExpr(phi), lambda: QFloatExpr(lambda), target)
    }

    @discardableResult
    public mutating func cz(_ control: Int, _ target: Int) throws -> QuantumCircuit {
        try applyValidated(.cz(control: control, target: target))
        return self
    }

    @discardableResult
    public mutating func swap(_ q1: Int, _ q2: Int) throws -> QuantumCircuit {
        try applyValidated(.swap(q1: q1, q2: q2))
        return self
    }

    @discardableResult
    public mutating func id(_ target: Int) throws -> QuantumCircuit {
        try applyValidated(.id(target: target))
        return self
    }

    @discardableResult
    public mutating func barrier(_ qubits: [Int] = []) throws -> QuantumCircuit {
        try applyValidated(.barrier(qubits: qubits))
        return self
    }

    @discardableResult
    public mutating func delay(duration: QFloat, _ qubit: Int) throws -> QuantumCircuit {
        try applyValidated(.delay(duration: duration, qubit: qubit))
        return self
    }

    @discardableResult
    public mutating func iswap(_ q1: Int, _ q2: Int) throws -> QuantumCircuit {
        try applyValidated(.iswap(q1: q1, q2: q2))
        return self
    }

    @discardableResult
    public mutating func ecr(control: Int, target: Int) throws -> QuantumCircuit {
        try applyValidated(.ecr(control: control, target: target))
        return self
    }

    @discardableResult
    public mutating func rxx(theta: QFloatExpr, _ q1: Int, _ q2: Int) throws -> QuantumCircuit {
        try applyValidated(.rxx(theta: theta, q1: q1, q2: q2))
        return self
    }

    @discardableResult
    public mutating func rxx(theta: QFloat, _ q1: Int, _ q2: Int) throws -> QuantumCircuit {
        try rxx(theta: QFloatExpr(theta), q1, q2)
    }

    @discardableResult
    public mutating func ryy(theta: QFloatExpr, _ q1: Int, _ q2: Int) throws -> QuantumCircuit {
        try applyValidated(.ryy(theta: theta, q1: q1, q2: q2))
        return self
    }

    @discardableResult
    public mutating func ryy(theta: QFloat, _ q1: Int, _ q2: Int) throws -> QuantumCircuit {
        try ryy(theta: QFloatExpr(theta), q1, q2)
    }

    @discardableResult
    public mutating func rzz(theta: QFloatExpr, _ q1: Int, _ q2: Int) throws -> QuantumCircuit {
        try applyValidated(.rzz(theta: theta, q1: q1, q2: q2))
        return self
    }

    @discardableResult
    public mutating func rzz(theta: QFloat, _ q1: Int, _ q2: Int) throws -> QuantumCircuit {
        try rzz(theta: QFloatExpr(theta), q1, q2)
    }

    @discardableResult
    public mutating func dcx(_ q1: Int, _ q2: Int) throws -> QuantumCircuit {
        try applyValidated(.dcx(q1: q1, q2: q2))
        return self
    }

    @discardableResult
    public mutating func cswap(control: Int, _ q1: Int, _ q2: Int) throws -> QuantumCircuit {
        try applyValidated(.cswap(control: control, q1: q1, q2: q2))
        return self
    }

    @discardableResult
    public mutating func crx(theta: QFloatExpr, control: Int, target: Int) throws -> QuantumCircuit {
        try applyValidated(.crx(theta: theta, control: control, target: target))
        return self
    }

    @discardableResult
    public mutating func crx(theta: QFloat, control: Int, target: Int) throws -> QuantumCircuit {
        try crx(theta: QFloatExpr(theta), control: control, target: target)
    }

    @discardableResult
    public mutating func cry(theta: QFloatExpr, control: Int, target: Int) throws -> QuantumCircuit {
        try applyValidated(.cry(theta: theta, control: control, target: target))
        return self
    }

    @discardableResult
    public mutating func cry(theta: QFloat, control: Int, target: Int) throws -> QuantumCircuit {
        try cry(theta: QFloatExpr(theta), control: control, target: target)
    }

    @discardableResult
    public mutating func crz(theta: QFloatExpr, control: Int, target: Int) throws -> QuantumCircuit {
        try applyValidated(.crz(theta: theta, control: control, target: target))
        return self
    }

    @discardableResult
    public mutating func crz(theta: QFloat, control: Int, target: Int) throws -> QuantumCircuit {
        try crz(theta: QFloatExpr(theta), control: control, target: target)
    }

    @discardableResult
    public mutating func cp(theta: QFloatExpr, control: Int, target: Int) throws -> QuantumCircuit {
        try applyValidated(.cp(theta: theta, control: control, target: target))
        return self
    }

    @discardableResult
    public mutating func cp(theta: QFloat, control: Int, target: Int) throws -> QuantumCircuit {
        try cp(theta: QFloatExpr(theta), control: control, target: target)
    }

    @discardableResult
    public mutating func mcx(controls: [Int], target: Int) throws -> QuantumCircuit {
        try applyValidated(.mcx(controls: controls, target: target))
        return self
    }

    @discardableResult
    public mutating func mcz(controls: [Int], target: Int) throws -> QuantumCircuit {
        try applyValidated(.mcz(controls: controls, target: target))
        return self
    }

    @discardableResult
    public mutating func x(_ target: Int) throws -> QuantumCircuit {
        try applyValidated(.x(target: target))
        return self
    }

    @discardableResult
    public mutating func y(_ target: Int) throws -> QuantumCircuit {
        try applyValidated(.y(target: target))
        return self
    }

    @discardableResult
    public mutating func z(_ target: Int) throws -> QuantumCircuit {
        try applyValidated(.z(target: target))
        return self
    }

    @discardableResult
    public mutating func measure(qubits: [Int]) throws -> QuantumCircuit {
        try applyValidated(.measure(MeasureSpec(qubits: qubits)))
        return self
    }

    @discardableResult
    public mutating func measure(
        qubits: [Int],
        classicalRegister: Int,
        classicalBitOffset: Int = 0
    ) throws -> QuantumCircuit {
        try applyValidated(
            .measure(
                MeasureSpec(
                    qubits: qubits,
                    classicalRegister: classicalRegister,
                    classicalBitOffset: classicalBitOffset
                )
            )
        )
        return self
    }

    @discardableResult
    public mutating func measure(_ qubit: Int) throws -> QuantumCircuit {
        try measure(qubits: [qubit])
    }

    @discardableResult
    public mutating func unitary1(matrix: [ComplexAmplitude], target: Int) throws -> QuantumCircuit {
        try applyValidated(.unitary1(matrix: matrix, target: target))
        return self
    }

    @discardableResult
    public mutating func initialize(
        qubits: [Int],
        amplitudes: [ComplexAmplitude]
    ) throws -> QuantumCircuit {
        try applyValidated(.initialize(qubits: qubits, amplitudes: amplitudes))
        return self
    }

    /// Prepares a computational-basis product state |index⟩ on the full register
    /// (qubit 0 = LSB of `index`).
    @discardableResult
    public mutating func initializeComputationalBasis(_ index: Int) throws -> QuantumCircuit {
        let stateCount = 1 << qubitCount
        guard index >= 0, index < stateCount else {
            throw QuantumCircuitError.invalidAlgorithmParameter(
                reason: "computational basis index \(index) is outside 0..<\(stateCount)"
            )
        }
        var amplitudes = Array(
            repeating: ComplexAmplitude(real: 0, imaginary: 0),
            count: stateCount
        )
        amplitudes[index] = ComplexAmplitude(real: 1, imaginary: 0)
        return try initialize(qubits: Array(0..<qubitCount), amplitudes: amplitudes)
    }

    /// Prepares a computational-basis bitstring on the listed qubits (leftmost = highest index
    /// in `bits`, matching MSB-first bitstring convention for that subset).
    @discardableResult
    public mutating func initializeProductState(qubits: [Int], bits: [Int]) throws -> QuantumCircuit {
        guard qubits.count == bits.count else {
            throw QuantumCircuitError.invalidAlgorithmParameter(
                reason: "product-state bits count must match qubits count"
            )
        }
        guard bits.allSatisfy({ $0 == 0 || $0 == 1 }) else {
            throw QuantumCircuitError.invalidAlgorithmParameter(
                reason: "product-state bits must be 0 or 1"
            )
        }
        var index = 0
        for (position, bit) in bits.enumerated() where bit == 1 {
            // bits array is MSB-first for the listed qubits order: bits[0] is highest qubit index weight
            let weight = qubits.count - 1 - position
            index |= bit << weight
        }
        // Map subset basis index into amplitudes on those qubits
        let count = 1 << qubits.count
        var amplitudes = Array(
            repeating: ComplexAmplitude(real: 0, imaginary: 0),
            count: count
        )
        amplitudes[index] = ComplexAmplitude(real: 1, imaginary: 0)
        return try initialize(qubits: qubits, amplitudes: amplitudes)
    }

    @discardableResult
    public mutating func customUnitary(
        matrix: [ComplexAmplitude],
        qubits: [Int]
    ) throws -> QuantumCircuit {
        try applyValidated(.customUnitary(matrix: matrix, qubits: qubits))
        return self
    }

    @discardableResult
    public mutating func c_if(
        classicalRegister: Int,
        equals expectedValue: Int,
        apply gate: Gate
    ) throws -> QuantumCircuit {
        try applyValidated(
            .c_if(
                classicalRegister: classicalRegister,
                expectedValue: expectedValue,
                gate: gate
            )
        )
        return self
    }

    @discardableResult
    public mutating func c_if(
        classicalRegister: Int,
        equals expectedValue: Int,
        x target: Int
    ) throws -> QuantumCircuit {
        try c_if(classicalRegister: classicalRegister, equals: expectedValue, apply: .x(target: target))
    }

    @discardableResult
    public mutating func reset(_ qubit: Int) throws -> QuantumCircuit {
        try applyValidated(.reset(qubit: qubit))
        return self
    }
}
