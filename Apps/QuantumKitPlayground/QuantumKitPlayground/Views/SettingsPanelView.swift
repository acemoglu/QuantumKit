import SwiftUI
import QuantumKit

struct SettingsPanelView: View {
    @EnvironmentObject private var viewModel: PlaygroundViewModel
    @Environment(\.isPhoneLayout) private var isPhoneLayout

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 12) {
                devicePicker
                shotsRow
                seedRow
                renormRow
                #if os(macOS)
                Button {
                    viewModel.isPresentingHelp = true
                } label: {
                    Label("How to Use", systemImage: "questionmark.circle")
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                #endif
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        } label: {
            Label("Run Settings", systemImage: "slider.horizontal.3")
        }
    }

    private var devicePicker: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Device")
                .font(.subheadline.weight(.semibold))
            Picker("Device", selection: $viewModel.settings.devicePreference) {
                Text("Automatic").tag(SimulationDevicePreference.automatic)
                Text("Metal").tag(SimulationDevicePreference.metal)
                Text("CPU").tag(SimulationDevicePreference.cpu)
            }
            .pickerStyle(.segmented)
            .labelsHidden()
        }
    }

    private var shotsRow: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Shots")
                .font(.subheadline.weight(.semibold))
            HStack(spacing: 8) {
                TextField("Shots", value: shotsBinding, format: IntegerFormatStyle<Int>().grouping(.never))
                    .textFieldStyle(.roundedBorder)
                    .font(.body.monospaced())
                    .frame(maxWidth: 120)
                    #if os(iOS)
                    .keyboardType(.numberPad)
                    #endif
                Stepper(
                    "Shots",
                    value: shotsBinding,
                    in: PlaygroundSettings.shotsRange,
                    step: 256
                )
                .labelsHidden()
                if !isPhoneLayout {
                    Text("\(viewModel.settings.shots)")
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }
        }
    }

    private var seedRow: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Seed")
                .font(.subheadline.weight(.semibold))
            if isPhoneLayout {
                TextField("Seed", text: seedTextBinding)
                    .textFieldStyle(.roundedBorder)
                    .font(.body.monospaced())
                    .disabled(viewModel.settings.useRandomSeed)
                    #if os(iOS)
                    .keyboardType(.numberPad)
                    #endif
                Toggle("Random seed", isOn: $viewModel.settings.useRandomSeed)
            } else {
                HStack(spacing: 8) {
                    TextField("Seed", text: seedTextBinding)
                        .textFieldStyle(.roundedBorder)
                        .font(.body.monospaced())
                        .frame(maxWidth: 160)
                        .disabled(viewModel.settings.useRandomSeed)
                    Toggle("Random seed", isOn: $viewModel.settings.useRandomSeed)
                        .toggleStyle(.checkboxCompatible)
                }
            }
        }
    }

    private var renormRow: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Renormalize every")
                .font(.subheadline.weight(.semibold))
            HStack(spacing: 8) {
                TextField(
                    "Gates",
                    value: renormBinding,
                    format: IntegerFormatStyle<Int>().grouping(.never)
                )
                .textFieldStyle(.roundedBorder)
                .font(.body.monospaced())
                .frame(maxWidth: 120)
                #if os(iOS)
                .keyboardType(.numberPad)
                #endif
                Text("gates")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                Spacer()
            }
            Text("Default 50. 0 turns periodic renormalization off.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var shotsBinding: Binding<Int> {
        Binding(
            get: { viewModel.settings.shots },
            set: { newValue in
                viewModel.settings.shots = min(
                    max(newValue, PlaygroundSettings.shotsRange.lowerBound),
                    PlaygroundSettings.shotsRange.upperBound
                )
            }
        )
    }

    private var renormBinding: Binding<Int> {
        Binding(
            get: { viewModel.settings.renormalizationInterval },
            set: { newValue in
                viewModel.settings.renormalizationInterval = min(
                    max(newValue, PlaygroundSettings.renormalizationIntervalRange.lowerBound),
                    PlaygroundSettings.renormalizationIntervalRange.upperBound
                )
            }
        )
    }

    private var seedTextBinding: Binding<String> {
        Binding(
            get: { String(viewModel.settings.seed) },
            set: { newValue in
                let digits = newValue.filter(\.isNumber)
                if let value = UInt64(digits) {
                    viewModel.settings.seed = value
                } else if digits.isEmpty {
                    viewModel.settings.seed = 0
                }
            }
        )
    }
}

private extension ToggleStyle where Self == CheckboxCompatibleToggleStyle {
    static var checkboxCompatible: CheckboxCompatibleToggleStyle { CheckboxCompatibleToggleStyle() }
}

private struct CheckboxCompatibleToggleStyle: ToggleStyle {
    func makeBody(configuration: Configuration) -> some View {
        #if os(macOS)
        Toggle(isOn: configuration.$isOn) { configuration.label }
            .toggleStyle(.checkbox)
        #else
        Toggle(isOn: configuration.$isOn) { configuration.label }
        #endif
    }
}

#Preview {
    SettingsPanelView()
        .environmentObject(PlaygroundViewModel.previewFactory())
        .padding()
}
