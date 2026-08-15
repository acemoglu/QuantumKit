import XCTest
import QuantumKit

extension QuantumKitTests {

    func testQuantumKitInfoVersionIsNonEmptySemVerLike() {
        let version = QuantumKitInfo.version
        XCTAssertFalse(version.isEmpty)
        // Smoke: MAJOR.MINOR.PATCH with non-empty components (pre-release suffixes allowed later).
        let core = version.split(separator: "-", maxSplits: 1, omittingEmptySubsequences: false).first.map(String.init) ?? version
        let parts = core.split(separator: ".", omittingEmptySubsequences: false)
        XCTAssertGreaterThanOrEqual(parts.count, 3, "expected at least major.minor.patch, got \(version)")
        for part in parts.prefix(3) {
            XCTAssertFalse(part.isEmpty)
            XCTAssertNotNil(UInt(part), "SemVer numeric component should parse: \(part) in \(version)")
        }
    }

    /// `0.2.0` is the honesty bump for H6c/H7b removals plus Pipelines/TRNGCollapse demotion
    /// documented on ``QuantumKitAPIPolicy``.
    func testQuantumKitInfoVersionIsPointTwoBreakingBump() {
        XCTAssertEqual(QuantumKitInfo.version, "0.2.0")
    }

    func testAPIPolicySymbolExistsForDocumentationAnchor() {
        // Public-module linkage smoke: policy enum is part of the exported surface
        // (DocC lists H6c/H7b removals and Pipelines / TRNGCollapse package-internal demotion).
        let _: QuantumKitAPIPolicy.Type = QuantumKitAPIPolicy.self
    }
}
