import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - Parser-level unsupported catalog

    func testOpenQASMUnsupportedDefcal() {
        assertUnsupportedKeyword(
            source: "OPENQASM 3.0;\ndefcal foo() q {}",
            feature: "defcal"
        )
    }

    func testOpenQASMUnsupportedFor() {
        assertUnsupportedKeyword(
            source: """
            OPENQASM 3.0;
            qubit q;
            for i in [0:1] { x q; }
            """,
            feature: "for"
        )
    }

    func testOpenQASMUnsupportedSwitch() {
        assertUnsupportedKeyword(
            source: """
            OPENQASM 3.0;
            bit c;
            switch (c) { case 0 {} }
            """,
            feature: "switch"
        )
    }

    func testOpenQASMUnsupportedBox() {
        assertUnsupportedKeyword(
            source: """
            OPENQASM 3.0;
            qubit q;
            box { x q; }
            """,
            feature: "box"
        )
    }

    func testOpenQASMUnsupportedExtern() {
        assertUnsupportedKeyword(
            source: "OPENQASM 3.0;\nextern foo();",
            feature: "extern"
        )
    }

    func testOpenQASMUnsupportedCal() {
        assertUnsupportedKeyword(
            source: "OPENQASM 3.0;\ncal {}",
            feature: "cal"
        )
    }

    func testOpenQASMUnsupportedGphase() {
        assertUnsupportedKeyword(
            source: "OPENQASM 3.0;\ngphase(0);",
            feature: "gphase"
        )
    }

    func testOpenQASMUnsupportedCtrlModifier() {
        assertUnsupportedKeyword(
            source: """
            OPENQASM 3.0;
            qubit[2] q;
            ctrl @ x q[0], q[1];
            """,
            feature: "ctrl"
        )
    }

    func testOpenQASMUnsupportedInvModifier() {
        assertUnsupportedKeyword(
            source: """
            OPENQASM 3.0;
            qubit q;
            inv @ x q;
            """,
            feature: "inv"
        )
    }

    func testOpenQASMUnsupportedPowModifier() {
        assertUnsupportedKeyword(
            source: """
            OPENQASM 3.0;
            qubit q;
            pow(2) @ x q;
            """,
            feature: "pow"
        )
    }

    func testOpenQASMUnsupportedAtModifierToken() {
        // Bare `@` after a non-modifier path still hard-errors.
        assertUnsupported(
            source: "OPENQASM 3.0;\n@ x q;",
            feature: OpenQASMUnsupportedFeature.atModifier.rawValue
        )
    }

    func testOpenQASMUnsupportedInput() {
        assertUnsupportedKeyword(
            source: "OPENQASM 3.0;\ninput angle[32] theta;",
            feature: "input"
        )
    }

    func testOpenQASMUnsupportedArray() {
        assertUnsupportedKeyword(
            source: "OPENQASM 3.0;\narray[int[8], 2] a;",
            feature: "array"
        )
    }

    func testOpenQASMUnsupportedStretch() {
        assertUnsupportedKeyword(
            source: "OPENQASM 3.0;\nstretch a;",
            feature: "stretch"
        )
    }

    func testOpenQASMUnsupportedDelay() {
        assertUnsupportedKeyword(
            source: "OPENQASM 3.0;\ndelay[0] q;",
            feature: "delay"
        )
    }

    func testOpenQASMUnsupportedFeatureCatalogNonEmpty() {
        XCTAssertFalse(OpenQASMUnsupported.parserRejectedKeywords.isEmpty)
        XCTAssertEqual(
            OpenQASMUnsupportedFeature.whileUnbounded.rawValue,
            "while"
        )
        XCTAssertTrue(
            OpenQASMUnsupported.whileMaxIterationsPragmaPrefix.contains("quantumkit")
        )
    }

    func testOpenQASMUnsupportedIncludesLocation() {
        let source = """
        OPENQASM 3.0;
        qubit q;
        defcal foo() q {}
        """
        XCTAssertThrowsError(try OpenQASM3Importer().`import`(source: source)) { error in
            guard case OpenQASMError.unsupported(let line, let column, let feature, _) = error else {
                return XCTFail("Expected unsupported, got \(error)")
            }
            XCTAssertEqual(feature, "defcal")
            XCTAssertEqual(line, 3)
            XCTAssertGreaterThan(column, 0)
        }
    }

    // MARK: - Helpers

    private func assertUnsupportedKeyword(
        source: String,
        feature: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        assertUnsupported(source: source, feature: feature, file: file, line: line)
    }

    private func assertUnsupported(
        source: String,
        feature: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertThrowsError(
            try OpenQASM3Importer().`import`(source: source),
            file: file,
            line: line
        ) { error in
            guard case OpenQASMError.unsupported(let locLine, let locColumn, let f, _) = error else {
                return XCTFail("Expected OpenQASMError.unsupported, got \(error)", file: file, line: line)
            }
            XCTAssertEqual(f, feature, file: file, line: line)
            XCTAssertGreaterThan(locLine, 0, file: file, line: line)
            XCTAssertGreaterThan(locColumn, 0, file: file, line: line)
        }
    }
}
