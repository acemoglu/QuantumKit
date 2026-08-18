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

    func testQuantumKitInfoVersionIsOneZero() {
        XCTAssertEqual(QuantumKitInfo.version, "1.0.0")
    }
}
