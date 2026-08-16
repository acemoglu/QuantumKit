import Foundation

/// Stable identifiers for OpenQASM features QuantumKit does **not** lower.
///
/// Use these ids in ``OpenQASMError/unsupported`` `feature` strings so tests and
/// tooling can match without depending on free-form messages.
///
/// ## Enforcement layers
/// 1. **Lexer / parser** — many QASM3 keywords (`defcal`, `for`, `switch`, `box`,
///    `ctrl`, …) throw ``OpenQASMError/unsupported`` as soon as the keyword is seen.
/// 2. **Importer** — constructs that parse into the AST but cannot map to ``Gate``
///    (e.g. unbounded `while` without a max-iterations bound) throw here.
/// 3. **Exporter** — IR ops with no QASM spelling (`unitary1`, …) throw on export.
public enum OpenQASMUnsupportedFeature: String, CaseIterable, Sendable, Hashable {
    // Pulse / calibration
    case defcal
    case cal
    case extern
    case delay
    case duration
    case stretch
    case durationof

    // Timing / boxing
    case box

    // Control-flow beyond the supported if / bounded-while subset
    case `for`
    case `switch`
    case `else`
    case `break`
    case `continue`
    case `return`
    case end

    // Subroutines / classical richness
    case `def`
    case input
    case output
    case `let`
    case `const`
    case array
    case complexClassicalAssign = "complex_classical_assign"

    // Gate modifiers / advanced unitaries
    case ctrl
    case negctrl
    case inv
    case pow
    case gphase
    case atModifier = "at_modifier"

    // Typed classical decls beyond bit/creg
    case angle
    case complex
    case bool
    case int
    case uint
    case float

    // Builtins we do not evaluate
    case sizeof
    case lengthof

    // Opaque / non-qelib1 includes (also used by importer)
    case opaque
    case include

    /// `while` without a positive max-iterations bound (options or pragma).
    case whileUnbounded = "while"

    /// Human-readable short description.
    public var summary: String {
        switch self {
        case .defcal: return "defcal calibration blocks"
        case .cal: return "cal blocks"
        case .extern: return "extern declarations"
        case .delay: return "delay / pulse timing"
        case .duration: return "duration types"
        case .stretch: return "stretch types"
        case .durationof: return "durationof()"
        case .box: return "box regions"
        case .for: return "for loops"
        case .switch: return "switch statements"
        case .else: return "else branches"
        case .break: return "break"
        case .continue: return "continue"
        case .return: return "return"
        case .end: return "end"
        case .def: return "def subroutines"
        case .input: return "input declarations"
        case .output: return "output declarations"
        case .let: return "let bindings"
        case .const: return "const declarations"
        case .array: return "array types"
        case .complexClassicalAssign: return "complex classical assignment"
        case .ctrl: return "ctrl@ modifiers"
        case .negctrl: return "negctrl@ modifiers"
        case .inv: return "inv@ modifiers"
        case .pow: return "pow@ modifiers"
        case .gphase: return "gphase"
        case .atModifier: return "@ gate modifiers"
        case .angle: return "angle types"
        case .complex: return "complex types"
        case .bool: return "bool types"
        case .int: return "int types"
        case .uint: return "uint types"
        case .float: return "float types"
        case .sizeof: return "sizeof()"
        case .lengthof: return "lengthof()"
        case .opaque: return "opaque gates"
        case .include: return "non-builtin include"
        case .whileUnbounded:
            return "while without maxIterations (set OpenQASM3ImporterOptions.defaultWhileMaxIterations or // @quantumkit.max_while_iterations N)"
        }
    }

    /// Builds ``OpenQASMError/unsupported`` at `location`.
    public func error(
        at location: SourceLocation,
        message: String? = nil
    ) -> OpenQASMError {
        .unsupported(
            line: location.line,
            column: location.column,
            feature: rawValue,
            message: message ?? "'\(rawValue)' is not supported by QuantumKit's OpenQASM front-end (\(summary))"
        )
    }
}

/// Catalog helpers for documentation and tests.
public enum OpenQASMUnsupported {
    /// Features the parser rejects immediately when the keyword appears at statement start.
    public static let parserRejectedKeywords: [OpenQASMUnsupportedFeature] = [
        .defcal, .cal, .extern, .delay, .box, .def, .switch, .for,
        .input, .output, .let, .const, .array, .duration, .stretch, .gphase,
        .ctrl, .negctrl, .inv, .pow, .else, .break, .continue, .return, .end,
        .angle, .complex, .bool, .int, .uint, .float,
        .durationof, .sizeof, .lengthof,
    ]

    /// Comment pragma recognized for bounded `while` → ``Gate/while_c`` lowering.
    ///
    /// Place on its own line immediately before a `while` statement:
    /// ```
    /// // @quantumkit.max_while_iterations 32
    /// while (c == 1) { x q[0]; }
    /// ```
    /// Blank lines and other `//` comments may appear between the pragma and `while`.
    /// A non-comment statement clears a pending pragma.
    ///
    /// Alternatively set ``OpenQASM3ImporterOptions/defaultWhileMaxIterations``.
    public static let whileMaxIterationsPragmaPrefix = "@quantumkit.max_while_iterations"
}

/// Scans OpenQASM source for `// @quantumkit.max_while_iterations N` comment pragmas.
///
/// Returns a map from the **1-based line of the `while` keyword** to `N`.
enum OpenQASMWhilePragmaScanner {
    /// Regex-free scan: `//` … `@quantumkit.max_while_iterations` … positive integer.
    static func scan(_ source: String) -> [Int: Int] {
        let lines = source.split(separator: "\n", omittingEmptySubsequences: false)
        var pending: Int?
        var result: [Int: Int] = [:]

        for (index, rawLine) in lines.enumerated() {
            let lineNumber = index + 1
            let line = String(rawLine)
            let trimmed = line.trimmingCharacters(in: .whitespaces)

            if let n = parsePragma(trimmed) {
                pending = n
                continue
            }

            if trimmed.isEmpty || trimmed.hasPrefix("//") {
                // Keep pending across blank / other comment lines.
                continue
            }

            if isWhileStatementStart(trimmed), let n = pending {
                result[lineNumber] = n
                pending = nil
                continue
            }

            // Any other code clears a dangling pragma.
            pending = nil
        }

        return result
    }

    private static func parsePragma(_ trimmed: String) -> Int? {
        guard trimmed.hasPrefix("//") else { return nil }
        var rest = trimmed.dropFirst(2)
        while rest.first?.isWhitespace == true {
            rest = rest.dropFirst()
        }
        let marker = OpenQASMUnsupported.whileMaxIterationsPragmaPrefix
        guard rest.hasPrefix(marker) else { return nil }
        rest = rest.dropFirst(marker.count)
        while rest.first?.isWhitespace == true {
            rest = rest.dropFirst()
        }
        var digits = ""
        while let ch = rest.first, ch.isASCII && ch.isNumber {
            digits.append(ch)
            rest = rest.dropFirst()
        }
        // Trailing junk after the integer is ignored if only whitespace / end.
        let trailing = rest.trimmingCharacters(in: .whitespaces)
        guard trailing.isEmpty || trailing.hasPrefix("//") else { return nil }
        guard let n = Int(digits), n > 0 else { return nil }
        return n
    }

    /// True when the first identifier on the line is `while` (after optional labels not used).
    private static func isWhileStatementStart(_ trimmed: String) -> Bool {
        var i = trimmed.startIndex
        guard i < trimmed.endIndex, trimmed[i].isLetter || trimmed[i] == "_" else {
            return false
        }
        let start = i
        i = trimmed.index(after: i)
        while i < trimmed.endIndex {
            let ch = trimmed[i]
            if ch.isLetter || ch.isNumber || ch == "_" {
                i = trimmed.index(after: i)
            } else {
                break
            }
        }
        return String(trimmed[start..<i]) == "while"
    }
}
