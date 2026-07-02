import Foundation

public enum ParameterBindingError: Error, Equatable {
    case missingBinding(for: String)
    case unboundParameters(Set<String>)
    case circuitContainsUnboundParameters(Set<String>)
}
