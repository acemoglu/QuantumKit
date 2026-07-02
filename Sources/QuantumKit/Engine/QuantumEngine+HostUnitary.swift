import Foundation

extension QuantumEngine {

    func applyHost1QUnitary(_ matrix: [ComplexAmplitude], target: Int, on state: StateVector) throws {
        guard matrix.count == 4 else {
            throw QuantumEngineError.functionNotFound("unitary1 matrix must contain 4 elements")
        }

        let realPointer = state.realBuffer.contents().assumingMemoryBound(to: QFloat.self)
        let imagPointer = state.imagBuffer.contents().assumingMemoryBound(to: QFloat.self)
        let bit = 1 << target

        var updatedReal = Array(repeating: QFloat(0), count: state.stateCount)
        var updatedImag = Array(repeating: QFloat(0), count: state.stateCount)

        for index in 0..<state.stateCount {
            let partner = index ^ bit
            if index > partner { continue }

            let aReal = realPointer[index]
            let aImag = imagPointer[index]
            let bReal = realPointer[partner]
            let bImag = imagPointer[partner]

            let m00 = matrix[0]
            let m01 = matrix[1]
            let m10 = matrix[2]
            let m11 = matrix[3]

            let outAReal = (m00.real * aReal) - (m00.imaginary * aImag)
                + (m01.real * bReal) - (m01.imaginary * bImag)
            let outAImag = (m00.real * aImag) + (m00.imaginary * aReal)
                + (m01.real * bImag) + (m01.imaginary * bReal)
            let outBReal = (m10.real * aReal) - (m10.imaginary * aImag)
                + (m11.real * bReal) - (m11.imaginary * bImag)
            let outBImag = (m10.real * aImag) + (m10.imaginary * aReal)
                + (m11.real * bImag) + (m11.imaginary * bReal)

            updatedReal[index] = outAReal
            updatedImag[index] = outAImag
            updatedReal[partner] = outBReal
            updatedImag[partner] = outBImag
        }

        for index in 0..<state.stateCount {
            realPointer[index] = updatedReal[index]
            imagPointer[index] = updatedImag[index]
        }
    }
}
