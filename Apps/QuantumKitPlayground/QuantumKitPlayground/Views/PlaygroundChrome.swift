import SwiftUI

enum PlaygroundChrome {
    static let cornerRadius: CGFloat = 10
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
                Text("QuantumKit Playground")
                    .font(.footnote.monospaced())
                    .foregroundStyle(.tertiary)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.bar)
    }
}
