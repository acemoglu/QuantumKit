import Foundation

/// Exact complex scalar used for analytic unitary verification.
struct UnitaryComplex: Equatable {
    var re: Double
    var im: Double

    static let zero = UnitaryComplex(re: 0, im: 0)
    static let one = UnitaryComplex(re: 1, im: 0)

    static func + (lhs: UnitaryComplex, rhs: UnitaryComplex) -> UnitaryComplex {
        UnitaryComplex(re: lhs.re + rhs.re, im: lhs.im + rhs.im)
    }

    static func * (lhs: UnitaryComplex, rhs: UnitaryComplex) -> UnitaryComplex {
        UnitaryComplex(
            re: (lhs.re * rhs.re) - (lhs.im * rhs.im),
            im: (lhs.re * rhs.im) + (lhs.im * rhs.re)
        )
    }
}

/// Row-major `dimension × dimension` unitary used for transpiler equivalence checks.
struct UnitaryMatrix: Equatable {
    let dimension: Int
    var elements: [UnitaryComplex]

    init(dimension: Int, elements: [UnitaryComplex]) {
        precondition(elements.count == dimension * dimension)
        self.dimension = dimension
        self.elements = elements
    }

    static func identity(_ dimension: Int) -> UnitaryMatrix {
        var elements = Array(repeating: UnitaryComplex.zero, count: dimension * dimension)
        for index in 0..<dimension {
            elements[(index * dimension) + index] = .one
        }
        return UnitaryMatrix(dimension: dimension, elements: elements)
    }

    subscript(row: Int, col: Int) -> UnitaryComplex {
        get { elements[(row * dimension) + col] }
        set { elements[(row * dimension) + col] = newValue }
    }

    func multiplied(by other: UnitaryMatrix) -> UnitaryMatrix {
        precondition(dimension == other.dimension)
        var product = Array(repeating: UnitaryComplex.zero, count: dimension * dimension)
        for row in 0..<dimension {
            for col in 0..<dimension {
                var sum = UnitaryComplex.zero
                for k in 0..<dimension {
                    sum = sum + (self[row, k] * other[k, col])
                }
                product[(row * dimension) + col] = sum
            }
        }
        return UnitaryMatrix(dimension: dimension, elements: product)
    }

    func isApproximatelyEqual(to other: UnitaryMatrix, tolerance: Double) -> Bool {
        guard dimension == other.dimension else { return false }
        for index in elements.indices {
            let deltaRe = abs(elements[index].re - other.elements[index].re)
            let deltaIm = abs(elements[index].im - other.elements[index].im)
            if deltaRe > tolerance || deltaIm > tolerance {
                return false
            }
        }
        return true
    }

    func scaled(by phase: UnitaryComplex) -> UnitaryMatrix {
        UnitaryMatrix(
            dimension: dimension,
            elements: elements.map { $0 * phase }
        )
    }
}

enum CircuitUnitary {

    private static func literalAngle(_ expr: QFloatExpr) throws -> QFloat {
        try expr.requireLiteral()
    }

    static func build(circuit: QuantumCircuit) throws -> UnitaryMatrix {
        guard circuit.isUnitaryOnly else {
            throw QuantumCircuitError.circuitNotUnitary
        }

        let dimension = 1 << circuit.qubitCount
        var product = UnitaryMatrix.identity(dimension)
        for gate in circuit.gates {
            let local = try matrix(for: gate, qubitCount: circuit.qubitCount)
            product = local.multiplied(by: product)
        }
        return product
    }

    static func areEquivalent(
        _ lhs: QuantumCircuit,
        _ rhs: QuantumCircuit,
        tolerance: Double = 1e-4
    ) throws -> Bool {
        guard lhs.qubitCount == rhs.qubitCount else {
            throw TranspilerError.mismatchedQubitCounts(
                original: lhs.qubitCount,
                transpiled: rhs.qubitCount
            )
        }

        let left = try build(circuit: lhs)
        let right = try build(circuit: rhs)
        return areUnitarilyEquivalent(left, right, tolerance: tolerance)
    }

    static func areUnitarilyEquivalent(
        _ lhs: UnitaryMatrix,
        _ rhs: UnitaryMatrix,
        tolerance: Double
    ) -> Bool {
        guard lhs.dimension == rhs.dimension else { return false }
        if lhs.isApproximatelyEqual(to: rhs, tolerance: tolerance) {
            return true
        }

        guard let phase = globalPhaseAligning(lhs, to: rhs, tolerance: tolerance) else {
            return false
        }
        return lhs.isApproximatelyEqual(to: rhs.scaled(by: phase), tolerance: tolerance)
    }

    private static func globalPhaseAligning(
        _ lhs: UnitaryMatrix,
        to rhs: UnitaryMatrix,
        tolerance: Double
    ) -> UnitaryComplex? {
        var referencePhase: UnitaryComplex?
        for index in lhs.elements.indices {
            let right = rhs.elements[index]
            let magnitude = hypot(right.re, right.im)
            if magnitude < tolerance { continue }

            let ratio = divide(lhs.elements[index], right)
            if let referencePhase {
                if !approximatelyEqual(ratio, referencePhase, tolerance: tolerance) {
                    return nil
                }
            } else {
                referencePhase = ratio
            }
        }
        return referencePhase
    }

    private static func divide(_ lhs: UnitaryComplex, _ rhs: UnitaryComplex) -> UnitaryComplex {
        let denominator = (rhs.re * rhs.re) + (rhs.im * rhs.im)
        return UnitaryComplex(
            re: ((lhs.re * rhs.re) + (lhs.im * rhs.im)) / denominator,
            im: ((lhs.im * rhs.re) - (lhs.re * rhs.im)) / denominator
        )
    }

    private static func approximatelyEqual(
        _ lhs: UnitaryComplex,
        _ rhs: UnitaryComplex,
        tolerance: Double
    ) -> Bool {
        abs(lhs.re - rhs.re) <= tolerance && abs(lhs.im - rhs.im) <= tolerance
    }

    private static func matrix(for gate: Gate, qubitCount: Int) throws -> UnitaryMatrix {
        switch gate {
        case .h(let target):
            let v = 1.0 / sqrt(2.0)
            return embed(
                u00: .init(re: v, im: 0), u01: .init(re: v, im: 0),
                u10: .init(re: v, im: 0), u11: .init(re: -v, im: 0),
                controlMask: 0, target: target, qubitCount: qubitCount
            )

        case .x(let target):
            return embed(
                u00: .zero, u01: .one,
                u10: .one, u11: .zero,
                controlMask: 0, target: target, qubitCount: qubitCount
            )

        case .y(let target):
            return embed(
                u00: .zero, u01: .init(re: 0, im: -1),
                u10: .init(re: 0, im: 1), u11: .zero,
                controlMask: 0, target: target, qubitCount: qubitCount
            )

        case .z(let target):
            return embed(
                u00: .one, u01: .zero,
                u10: .zero, u11: .init(re: -1, im: 0),
                controlMask: 0, target: target, qubitCount: qubitCount
            )

        case .s(let target):
            return embed(
                u00: .one, u01: .zero,
                u10: .zero, u11: .init(re: 0, im: 1),
                controlMask: 0, target: target, qubitCount: qubitCount
            )

        case .t(let target):
            let v = 1.0 / sqrt(2.0)
            return embed(
                u00: .one, u01: .zero,
                u10: .zero, u11: .init(re: v, im: v),
                controlMask: 0, target: target, qubitCount: qubitCount
            )

        case .sdg(let target):
            return embed(
                u00: .one, u01: .zero,
                u10: .zero, u11: .init(re: 0, im: -1),
                controlMask: 0, target: target, qubitCount: qubitCount
            )

        case .tdg(let target):
            let v = 1.0 / sqrt(2.0)
            return embed(
                u00: .one, u01: .zero,
                u10: .zero, u11: .init(re: v, im: -v),
                controlMask: 0, target: target, qubitCount: qubitCount
            )

        case .sx(let target):
            return embed(
                u00: .init(re: 0.5, im: 0.5), u01: .init(re: 0.5, im: -0.5),
                u10: .init(re: 0.5, im: -0.5), u11: .init(re: 0.5, im: 0.5),
                controlMask: 0, target: target, qubitCount: qubitCount
            )

        case .sxdg(let target):
            return embed(
                u00: .init(re: 0.5, im: -0.5), u01: .init(re: 0.5, im: 0.5),
                u10: .init(re: 0.5, im: 0.5), u11: .init(re: 0.5, im: -0.5),
                controlMask: 0, target: target, qubitCount: qubitCount
            )

        case .p(let theta, let target):
            return try rzMatrix(theta: theta, target: target, qubitCount: qubitCount)

        case .rx(let theta, let target):
            let half = Double(try literalAngle(theta)) / 2.0
            let c = cos(half)
            let s = sin(half)
            return embed(
                u00: .init(re: c, im: 0), u01: .init(re: 0, im: -s),
                u10: .init(re: 0, im: -s), u11: .init(re: c, im: 0),
                controlMask: 0, target: target, qubitCount: qubitCount
            )

        case .ry(let theta, let target):
            let half = Double(try literalAngle(theta)) / 2.0
            let c = cos(half)
            let s = sin(half)
            return embed(
                u00: .init(re: c, im: 0), u01: .init(re: -s, im: 0),
                u10: .init(re: s, im: 0), u11: .init(re: c, im: 0),
                controlMask: 0, target: target, qubitCount: qubitCount
            )

        case .rz(let theta, let target):
            return try rzMatrix(theta: theta, target: target, qubitCount: qubitCount)

        case .u(let theta, let phi, let lambda, let target):
            let thetaValue = try literalAngle(theta)
            let phiValue = try literalAngle(phi)
            let lambdaValue = try literalAngle(lambda)
            let half = Double(thetaValue) / 2.0
            let c = cos(half)
            let s = sin(half)
            let eLambda = UnitaryComplex(re: cos(Double(lambdaValue)), im: sin(Double(lambdaValue)))
            let ePhi = UnitaryComplex(re: cos(Double(phiValue)), im: sin(Double(phiValue)))
            let ePhiLambda = UnitaryComplex(
                re: cos(Double(phiValue + lambdaValue)),
                im: sin(Double(phiValue + lambdaValue))
            )
            return embed(
                u00: .init(re: c, im: 0),
                u01: UnitaryComplex(re: -s, im: 0) * eLambda,
                u10: UnitaryComplex(re: s, im: 0) * ePhi,
                u11: UnitaryComplex(re: c, im: 0) * ePhiLambda,
                controlMask: 0, target: target, qubitCount: qubitCount
            )

        case .cx(let control, let target):
            return embed(
                u00: .zero, u01: .one,
                u10: .one, u11: .zero,
                controlMask: 1 << control, target: target, qubitCount: qubitCount
            )

        case .cz(let control, let target):
            return embed(
                u00: .one, u01: .zero,
                u10: .zero, u11: .init(re: -1, im: 0),
                controlMask: 1 << control, target: target, qubitCount: qubitCount
            )

        case .swap(let q1, let q2):
            return try matrix(for: .cx(control: q1, target: q2), qubitCount: qubitCount)
                .multiplied(by: try matrix(for: .cx(control: q2, target: q1), qubitCount: qubitCount))
                .multiplied(by: try matrix(for: .cx(control: q1, target: q2), qubitCount: qubitCount))

        case .id:
            return UnitaryMatrix.identity(1 << qubitCount)

        case .iswap, .ecr, .rxx, .ryy, .rzz, .dcx, .cswap:
            return try matrixFromDecomposition(gate, qubitCount: qubitCount)

        case .ccx(let control1, let control2, let target):
            return embed(
                u00: .zero, u01: .one,
                u10: .one, u11: .zero,
                controlMask: (1 << control1) | (1 << control2),
                target: target,
                qubitCount: qubitCount
            )

        case .mcx(let controls, let target):
            let mask = controls.reduce(0) { $0 | (1 << $1) }
            return embed(
                u00: .zero, u01: .one,
                u10: .one, u11: .zero,
                controlMask: mask,
                target: target,
                qubitCount: qubitCount
            )

        case .mcz(let controls, let target):
            let mask = controls.reduce(0) { $0 | (1 << $1) }
            return embed(
                u00: .one, u01: .zero,
                u10: .zero, u11: .init(re: -1, im: 0),
                controlMask: mask,
                target: target,
                qubitCount: qubitCount
            )

        case .crx(let theta, let control, let target):
            return try matrix(for: .rz(theta: QFloatExpr(-QFloat(Double.pi / 2)), target: target), qubitCount: qubitCount)
                .multiplied(by: try matrix(for: .cx(control: control, target: target), qubitCount: qubitCount))
                .multiplied(by: try matrix(for: .rz(theta: theta, target: target), qubitCount: qubitCount))
                .multiplied(by: try matrix(for: .cx(control: control, target: target), qubitCount: qubitCount))
                .multiplied(by: try matrix(for: .rz(theta: QFloatExpr(QFloat(Double.pi / 2)), target: target), qubitCount: qubitCount))

        case .cry(let theta, let control, let target):
            return try matrix(for: .ry(theta: theta / 2, target: target), qubitCount: qubitCount)
                .multiplied(by: try matrix(for: .cx(control: control, target: target), qubitCount: qubitCount))
                .multiplied(by: try matrix(for: .ry(theta: -theta / 2, target: target), qubitCount: qubitCount))
                .multiplied(by: try matrix(for: .cx(control: control, target: target), qubitCount: qubitCount))

        case .crz(let theta, let control, let target):
            return try matrix(for: .rz(theta: theta / 2, target: target), qubitCount: qubitCount)
                .multiplied(by: try matrix(for: .cx(control: control, target: target), qubitCount: qubitCount))
                .multiplied(by: try matrix(for: .rz(theta: -theta / 2, target: target), qubitCount: qubitCount))
                .multiplied(by: try matrix(for: .cx(control: control, target: target), qubitCount: qubitCount))

        case .cp(let theta, let control, let target):
            return try matrix(for: .crz(theta: theta, control: control, target: target), qubitCount: qubitCount)

        case .unitary1(let matrix, let target):
            guard matrix.count == 4 else {
                throw QuantumCircuitError.invalidAlgorithmParameter(
                    reason: "unitary1 requires exactly four complex amplitudes"
                )
            }
            return embed(
                u00: .init(re: Double(matrix[0].real), im: Double(matrix[0].imaginary)),
                u01: .init(re: Double(matrix[1].real), im: Double(matrix[1].imaginary)),
                u10: .init(re: Double(matrix[2].real), im: Double(matrix[2].imaginary)),
                u11: .init(re: Double(matrix[3].real), im: Double(matrix[3].imaginary)),
                controlMask: 0, target: target, qubitCount: qubitCount
            )

        case .customUnitary(let matrix, let qubits):
            guard qubits.count == 1, let target = qubits.first, matrix.count == 4 else {
                throw QuantumCircuitError.invalidAlgorithmParameter(
                    reason: "customUnitary circuit unitary builder supports single-qubit matrices only"
                )
            }
            return embed(
                u00: .init(re: Double(matrix[0].real), im: Double(matrix[0].imaginary)),
                u01: .init(re: Double(matrix[1].real), im: Double(matrix[1].imaginary)),
                u10: .init(re: Double(matrix[2].real), im: Double(matrix[2].imaginary)),
                u11: .init(re: Double(matrix[3].real), im: Double(matrix[3].imaginary)),
                controlMask: 0, target: target, qubitCount: qubitCount
            )

        case .initialize, .measure, .reset, .c_if, .barrier, .delay:
            throw QuantumCircuitError.circuitNotUnitary
        }
    }

    private static func matrixFromDecomposition(_ gate: Gate, qubitCount: Int) throws -> UnitaryMatrix {
        let pieces = try GateDecomposition.expand(gate)
        var product = UnitaryMatrix.identity(1 << qubitCount)
        for piece in pieces {
            let local = try matrix(for: piece, qubitCount: qubitCount)
            product = local.multiplied(by: product)
        }
        return product
    }

    private static func rzMatrix(theta: QFloatExpr, target: Int, qubitCount: Int) throws -> UnitaryMatrix {
        let half = Double(try literalAngle(theta)) / 2.0
        return embed(
            u00: .init(re: cos(half), im: -sin(half)),
            u01: .zero,
            u10: .zero,
            u11: .init(re: cos(half), im: sin(half)),
            controlMask: 0,
            target: target,
            qubitCount: qubitCount
        )
    }

    private static func embed(
        u00: UnitaryComplex,
        u01: UnitaryComplex,
        u10: UnitaryComplex,
        u11: UnitaryComplex,
        controlMask: Int,
        target: Int,
        qubitCount: Int
    ) -> UnitaryMatrix {
        let dimension = 1 << qubitCount
        var matrix = UnitaryMatrix.identity(dimension)
        let passiveMask = ((1 << qubitCount) - 1) & ~(1 << target) & ~controlMask

        for col in 0..<dimension {
            for row in 0..<dimension {
                if (row & passiveMask) != (col & passiveMask) {
                    matrix[row, col] = .zero
                    continue
                }

                if controlMask != 0 {
                    if (col & controlMask) != controlMask {
                        matrix[row, col] = row == col ? .one : .zero
                        continue
                    }
                    if (row & controlMask) != controlMask {
                        matrix[row, col] = .zero
                        continue
                    }
                }

                let targetCol = (col >> target) & 1
                let targetRow = (row >> target) & 1
                matrix[row, col] = singleQubitElement(
                    row: targetRow,
                    col: targetCol,
                    u00: u00,
                    u01: u01,
                    u10: u10,
                    u11: u11
                )
            }
        }

        return matrix
    }

    private static func singleQubitElement(
        row: Int,
        col: Int,
        u00: UnitaryComplex,
        u01: UnitaryComplex,
        u10: UnitaryComplex,
        u11: UnitaryComplex
    ) -> UnitaryComplex {
        switch (row, col) {
        case (0, 0): return u00
        case (0, 1): return u01
        case (1, 0): return u10
        case (1, 1): return u11
        default: return .zero
        }
    }
}
