import SwiftUI

enum PlaygroundChrome {
    static let cornerRadius: CGFloat = 10
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
                .foregroundStyle(.white)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.headline)
                    .foregroundStyle(.white)
                Text(message)
                    .font(.callout)
                    .foregroundStyle(.white)
                    .textSelection(.enabled)
            }
            Spacer(minLength: 0)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.red, in: RoundedRectangle(cornerRadius: PlaygroundChrome.cornerRadius, style: .continuous))
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
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: PlaygroundChrome.cornerRadius, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: PlaygroundChrome.cornerRadius, style: .continuous)
                    .strokeBorder(Color.primary.opacity(0.12))
            )
    }
}

extension Color {
    static var playgroundEditorBackground: Color {
        #if os(macOS)
        Color(nsColor: .textBackgroundColor)
        #else
        Color(uiColor: .secondarySystemBackground)
        #endif
    }
}

struct StatusBarView: View {
    let message: String
    let isBusy: Bool
    var lastRunSummary: String? = nil

    var body: some View {
        HStack(spacing: 8) {
            if isBusy {
                ProgressView()
                    .controlSize(.small)
            }
            Text(message)
                .font(.footnote)
                .foregroundStyle(.secondary)
                .lineLimit(2)
            Spacer()
            if let lastRunSummary {
                Text(lastRunSummary)
                    .font(.footnote.monospaced())
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            } else {
                HStack(spacing: 6) {
                    QuantumKitMark(size: 14)
                    Text("QuantumKit")
                        .font(.footnote.monospaced())
                        .foregroundStyle(.tertiary)
                }
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.bar)
    }
}
