import SwiftUI

#if canImport(AppKit)
import AppKit
#endif
#if canImport(UIKit)
import UIKit
#endif

enum PlaygroundChrome {
    /// Matches the rounded CPU-die mark.
    static let cornerRadius: CGFloat = 14
    static let chipRadius: CGFloat = 8
}

private struct PhoneLayoutKey: EnvironmentKey {
    static let defaultValue = false
}

extension EnvironmentValues {
    /// True on iPhone (including landscape) and iPad compact width (slide over).
    var isPhoneLayout: Bool {
        get { self[PhoneLayoutKey.self] }
        set { self[PhoneLayoutKey.self] = newValue }
    }
}

/// In-app mark: CPU die + atom (same artwork as the app icon).
struct QuantumKitMark: View {
    var size: CGFloat = 22

    var body: some View {
        Image("QuantumKitLogo")
            .resizable()
            .interpolation(.high)
            .scaledToFit()
            .frame(width: size, height: size)
            .clipShape(RoundedRectangle(cornerRadius: size * 0.22, style: .continuous))
            .accessibilityLabel("QuantumKit")
    }
}

struct ErrorBannerView: View {
    let title: String
    let message: String

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.title3)
                .foregroundStyle(Color.quantumOnAccent)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.headline)
                    .foregroundStyle(Color.quantumOnAccent)
                Text(message)
                    .font(.callout)
                    .foregroundStyle(Color.quantumOnAccent)
                    .textSelection(.enabled)
            }
            Spacer(minLength: 0)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .foregroundStyle(Color.quantumOnAccent)
        .background(Color.quantumInk, in: RoundedRectangle(cornerRadius: PlaygroundChrome.cornerRadius, style: .continuous))
        .accessibilityElement(children: .combine)
    }
}

struct PlaygroundCard<Content: View>: View {
    let content: Content

    init(@ViewBuilder content: () -> Content) {
        self.content = content()
    }

    var body: some View {
        content
            .padding(10)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            .playgroundPanel()
    }
}

extension View {
    func playgroundPanel(enabled: Bool = true) -> some View {
        modifier(PlaygroundPanelChrome(enabled: enabled))
    }
}

private struct PlaygroundPanelChrome: ViewModifier {
    var enabled: Bool

    func body(content: Content) -> some View {
        if enabled {
            content
                .background(
                    Color.quantumCard,
                    in: RoundedRectangle(cornerRadius: PlaygroundChrome.cornerRadius, style: .continuous)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: PlaygroundChrome.cornerRadius, style: .continuous)
                        .strokeBorder(Color.quantumInk.opacity(0.10))
                )
        } else {
            content
        }
    }
}

extension Color {
    /// Logo wash: ice-white in light, cool charcoal in dark.
    static var quantumCanvas: Color {
        Color(light: Color(red: 0.957, green: 0.969, blue: 0.984),
              dark: Color(red: 0.067, green: 0.078, blue: 0.094))
    }

    static var quantumCard: Color {
        Color(light: Color(red: 1, green: 1, blue: 1),
              dark: Color(red: 0.110, green: 0.122, blue: 0.141))
    }

    static var quantumInk: Color {
        Color(light: Color(red: 0.070, green: 0.075, blue: 0.090),
              dark: Color(red: 0.957, green: 0.969, blue: 0.984))
    }

    /// Steel, not orange — pending placement still reads, stays on-brand.
    static var quantumPending: Color {
        Color(light: Color(red: 0.29, green: 0.46, blue: 0.58),
              dark: Color(red: 0.62, green: 0.78, blue: 0.89))
    }

    static var quantumOnAccent: Color {
        Color(light: .white, dark: Color(red: 0.070, green: 0.075, blue: 0.090))
    }

    static var playgroundEditorBackground: Color { quantumCard }

    fileprivate init(light: Color, dark: Color) {
        #if os(iOS)
        self.init(uiColor: UIColor { traits in
            traits.userInterfaceStyle == .dark ? UIColor(dark) : UIColor(light)
        })
        #elseif os(macOS)
        self.init(nsColor: NSColor(name: nil, dynamicProvider: { appearance in
            let isDark = appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
            return NSColor(isDark ? dark : light)
        }))
        #else
        self = light
        #endif
    }
}

struct DelayedHoverHint: ViewModifier {
    let text: String
    @Binding var activeHint: String?
    @State private var generation = 0

    func body(content: Content) -> some View {
        content
            .help(text)
            .onHover { hovering in
                generation += 1
                let token = generation
                if hovering {
                    Task { @MainActor in
                        try? await Task.sleep(for: .milliseconds(1200))
                        guard token == generation else { return }
                        activeHint = text
                    }
                } else if activeHint == text {
                    activeHint = nil
                }
            }
    }
}

extension View {
    func delayedHoverHint(_ text: String, activeHint: Binding<String?>) -> some View {
        modifier(DelayedHoverHint(text: text, activeHint: activeHint))
    }
}
