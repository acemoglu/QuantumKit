import Foundation

public enum ShorClassicalError: Error, Equatable {
    case invalidModulus(Int)
    case baseNotCoprimeToModulus(base: Int, modulus: Int)
}

/// Classical post-processing for Shor's period-finding and factoring step.
public struct ShorClassical: Sendable {

    public struct Analysis: Sendable, Equatable {
        public let recoveredPeriods: Set<Int>
        public let foundFactors: Set<Int>

        public init(recoveredPeriods: Set<Int>, foundFactors: Set<Int>) {
            self.recoveredPeriods = recoveredPeriods
            self.foundFactors = foundFactors
        }
    }

    /// Multiplicative order of `base` modulo `modulus` (smallest `r > 0` with `base^r ≡ 1`).
    public static func multiplicativeOrder(base: Int, modulus: Int) throws -> Int {
        try validateCoprimeBase(base, modulus: modulus)

        var power = 1
        var order = 0
        repeat {
            order += 1
            power = (power * base) % modulus
        } while power != 1
        return order
    }

    public static func modularPower(_ base: Int, exponent: Int, modulus: Int) -> Int {
        guard modulus > 0 else { return 0 }

        var result = 1
        var b = ((base % modulus) + modulus) % modulus
        var e = max(exponent, 0)

        while e > 0 {
            if e & 1 == 1 {
                result = (result * b) % modulus
            }
            b = (b * b) % modulus
            e >>= 1
        }

        return result
    }

    public static func greatestCommonDivisor(_ lhs: Int, _ rhs: Int) -> Int {
        var a = abs(lhs)
        var b = abs(rhs)
        while b != 0 {
            (a, b) = (b, a % b)
        }
        return a
    }

    /// Estimates the period `r` from a single counting-register measurement via continued fractions.
    public static func estimatePeriod(
        measurement: Int,
        controlQubitCount: Int,
        base: Int,
        modulus: Int
    ) -> Int? {
        guard measurement != 0, controlQubitCount > 0, modulus > 1 else { return nil }

        let denominatorLimit = modulus * modulus
        let phase = Double(measurement) / Double(1 << controlQubitCount)

        var x = phase
        var hPrev = 0
        var hCurr = 1
        var kPrev = 1
        var kCurr = 0

        for _ in 0..<32 {
            let coefficient = Int(floor(x))
            let hNext = coefficient * hCurr + hPrev
            let kNext = coefficient * kCurr + kPrev

            if kNext > 0,
               kNext <= denominatorLimit,
               modularPower(base, exponent: kNext, modulus: modulus) == 1 {
                return kNext
            }

            if x == Double(coefficient) { break }

            x = 1.0 / (x - Double(coefficient))
            hPrev = hCurr
            hCurr = hNext
            kPrev = kCurr
            kCurr = kNext
        }

        return nil
    }

    /// Factors `modulus` using an even period `r` with `base^r ≡ 1 (mod modulus)`.
    public static func factorsFromPeriod(
        base: Int,
        period: Int,
        modulus: Int
    ) -> (Int, Int)? {
        guard period % 2 == 0, modulus > 1 else { return nil }

        let halfPower = modularPower(base, exponent: period / 2, modulus: modulus)
        guard halfPower != 1, halfPower != modulus - 1 else { return nil }

        let factorA = greatestCommonDivisor(halfPower - 1, modulus)
        let factorB = greatestCommonDivisor(halfPower + 1, modulus)

        guard factorA > 1, factorA < modulus, factorB > 1, factorB < modulus else {
            return nil
        }

        return (min(factorA, factorB), max(factorA, factorB))
    }

    /// Runs period estimation and factoring over every distinct outcome in a shot histogram.
    public static func analyze(
        counts: ShotCounts,
        controlQubitCount: Int,
        base: Int,
        modulus: Int
    ) -> Analysis {
        var recoveredPeriods: Set<Int> = []
        var foundFactors: Set<Int> = []

        for (measurement, _) in counts.counts {
            guard let period = estimatePeriod(
                measurement: measurement,
                controlQubitCount: controlQubitCount,
                base: base,
                modulus: modulus
            ) else {
                continue
            }

            recoveredPeriods.insert(period)

            guard let (factorA, factorB) = factorsFromPeriod(
                base: base,
                period: period,
                modulus: modulus
            ) else {
                continue
            }

            foundFactors.insert(factorA)
            foundFactors.insert(factorB)
        }

        return Analysis(recoveredPeriods: recoveredPeriods, foundFactors: foundFactors)
    }

    private static func validateCoprimeBase(_ base: Int, modulus: Int) throws {
        guard modulus > 1 else {
            throw ShorClassicalError.invalidModulus(modulus)
        }

        let reduced = ((base % modulus) + modulus) % modulus
        guard reduced != 0, greatestCommonDivisor(reduced, modulus) == 1 else {
            throw ShorClassicalError.baseNotCoprimeToModulus(base: base, modulus: modulus)
        }
    }
}
