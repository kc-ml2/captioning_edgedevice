import SwiftUI

struct ContentView: View {
    @Environment(\.scenePhase) private var scenePhase
    @StateObject private var model = CaptionScreenModel()
    var body: some View {
        Group {
            if model.modelReady {
                cameraScreen
            } else {
                modelSetupScreen
            }
        }
        .task { model.sceneChanged(scenePhase); await model.prepareApp() }
        .onChange(of: scenePhase) { _, phase in model.sceneChanged(phase) }
        .preferredColorScheme(.dark)
    }

    private var cameraScreen: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            CameraPreview(session: model.camera.session).ignoresSafeArea()

            if model.camera.permissionDenied {
                cameraMessage(icon: "model.camera.fill", title: L10n.cameraPermissionTitle, message: L10n.cameraPermissionMessage)
            } else if model.camera.isUnavailable {
                #if targetEnvironment(simulator)
                cameraMessage(icon: "photo.fill", title: L10n.sampleModeTitle, message: L10n.sampleModeMessage)
                #else
                cameraMessage(icon: "model.camera.slash.fill", title: L10n.cameraUnavailableTitle, message: L10n.cameraUnavailableMessage)
                #endif
            }

            VStack(spacing: 0) {
                topBar
                Spacer()
                if let caption = model.caption { resultBubble(caption) }
                captionPanel
            }
        }
    }

    private var modelSetupScreen: some View {
        ZStack {
            LinearGradient(
                colors: [Color.black, Color(red: 0.12, green: 0.08, blue: 0.03)],
                startPoint: .top,
                endPoint: .bottom
            )
            .ignoresSafeArea()

            VStack(spacing: 0) {
                topBar
                Spacer()

                VStack(spacing: 22) {
                    Image(systemName: model.isInstallingModel ? "arrow.down.circle.fill" : "shippingbox.fill")
                        .font(.system(size: 58, weight: .medium))
                        .foregroundStyle(.orange)
                        .symbolEffect(.pulse, isActive: model.isInstallingModel)

                    VStack(spacing: 8) {
                        Text(model.isCheckingModel ? L10n.checkingModelTitle : L10n.prepareModelTitle)
                            .font(.title2.bold())
                        Text(setupDescription)
                            .font(.subheadline)
                            .foregroundStyle(.white.opacity(0.7))
                            .multilineTextAlignment(.center)
                            .lineSpacing(3)
                    }

                    if model.isInstallingModel {
                        VStack(spacing: 10) {
                            ProgressView(value: model.modelProgress)
                                .tint(.orange)
                            HStack {
                                Text(downloadProgressText)
                                Spacer()
                                Text(model.modelProgress, format: .percent.precision(.fractionLength(0)))
                            }
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.white.opacity(0.65))
                        }
                    }

                    if let status = model.modelStatus {
                        Text(status)
                            .font(.footnote.weight(.medium))
                            .foregroundStyle(.white.opacity(0.8))
                            .multilineTextAlignment(.center)
                    }

                    if let error = model.errorMessage {
                        Label(error, systemImage: "exclamationmark.triangle.fill")
                            .font(.footnote)
                            .foregroundStyle(.orange)
                            .multilineTextAlignment(.center)
                    }

                    setupButton
                }
                .padding(28)
                .background(.white.opacity(0.07), in: RoundedRectangle(cornerRadius: 28))
                .overlay {
                    RoundedRectangle(cornerRadius: 28)
                        .stroke(.white.opacity(0.1), lineWidth: 1)
                }
                .padding(.horizontal, 24)

                Spacer()

                Label(L10n.noCameraDuringDownload, systemImage: "model.camera.fill")
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.55))
                    .padding(.bottom, 24)
            }
        }
    }

    @ViewBuilder
    private var setupButton: some View {
        if model.isInstallingModel {
            Button(role: .cancel) { model.cancelInstallation() } label: {
                Label(L10n.cancelDownload, systemImage: "xmark.circle.fill")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .tint(.white)
        } else {
            Button { model.modelDownloadSize == nil ? model.retryModelCheck() : model.installModel() } label: {
                Label(
                    model.modelDownloadSize == nil ? L10n.retry : L10n.downloadModel,
                    systemImage: model.modelDownloadSize == nil ? "arrow.clockwise" : "icloud.and.arrow.down.fill"
                )
                .fontWeight(.semibold)
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .tint(.orange)
            .disabled(model.isCheckingModel)
        }
    }

    private var setupDescription: String {
        if model.isCheckingModel { return L10n.checkingInstalledModel }
        if let size = model.modelDownloadSize {
            return L10n.modelDownloadDescription(size: size)
        }
        return L10n.modelInfoUnavailable
    }

    private var downloadProgressText: String {
        let unit = DownloadSizeUnit.forTotalBytes(model.totalDownloadBytes)
        let completed = unit.format(model.downloadedBytes)
        let total = unit.format(model.totalDownloadBytes)
        return "\(completed) / \(total)"
    }

    private var topBar: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(verbatim: "SceneSense").font(.headline.weight(.bold))
                Text("On-Device Image Captioning").font(.caption2).foregroundStyle(.white.opacity(0.7))
            }
            Spacer()
            Button { model.speech.toggle(model.caption ?? "") } label: {
                Image(systemName: model.speech.outputVolume == 0 ? "speaker.slash.fill"
                      : model.speech.isSpeaking ? "speaker.wave.2.fill" : "speaker.fill")
                    .font(.system(size: 16, weight: .semibold))
                    .frame(width: 44, height: 44)
                    .background(.ultraThinMaterial, in: Circle())
            }
            .buttonStyle(.plain)
            .disabled(model.caption?.isEmpty != false)
            .accessibilityLabel(model.speech.isSpeaking ? Text("Stop reading") : Text("Read aloud"))
            .accessibilityValue(model.speech.outputVolume == 0 ? Text("Device volume is zero")
                                 : model.speech.isSpeaking ? Text("Reading caption") : Text("Ready to read"))
            Image(systemName: "lock.shield.fill")
                .font(.system(size: 16, weight: .semibold))
                .padding(10).background(.ultraThinMaterial, in: Circle())
                .accessibilityLabel(L10n.onDeviceProcessing)
        }
        .padding(.horizontal, 20).padding(.top, 8).padding(.bottom, 18)
        .background(LinearGradient(colors: [.black.opacity(0.75), .clear], startPoint: .top, endPoint: .bottom).ignoresSafeArea(edges: .top))
    }

    private var captionPanel: some View {
        VStack(spacing: 16) {
            Text(captionStatusText)
                .font(.subheadline.weight(.medium))
                .foregroundStyle(model.errorMessage == nil ? .white.opacity(0.9) : .orange)
                .multilineTextAlignment(.center)
                .padding(.horizontal)

            Button { model.generateCaption() } label: {
                ZStack {
                    Circle().stroke(.white.opacity(0.95), lineWidth: 4).frame(width: 82, height: 82)
                    Circle().fill(model.isCaptioning ? Color.orange : Color.white).frame(width: 66, height: 66)
                    Image(systemName: model.isCaptioning ? "ellipsis.message.fill" : "text.bubble.fill")
                        .font(.system(size: 29, weight: .semibold)).foregroundStyle(.black)
                        .symbolEffect(.pulse, isActive: model.isCaptioning)
                }
            }
            .buttonStyle(.plain).disabled(model.isCaptioning)
            .accessibilityLabel(L10n.generateCaption)
        }
        .frame(maxWidth: .infinity).padding(.top, 18).padding(.bottom, 18)
        .background(.black.opacity(0.72))
    }

    private var captionStatusText: String {
        if model.isCaptioning { return L10n.describingScene }
        if let error = model.errorMessage { return error }
        return L10n.captionPrompt
    }

    private func resultBubble(_ text: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            CaptionActions(caption: text, benchmark: model.captionBenchmark, sourceImage: model.captionSourceImage, speech: model.speech)
                .id(model.captionID)
            ScrollView {
                Text(text)
                    .font(.body.weight(.medium)).foregroundStyle(.black)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 18).padding(.vertical, 14)
            }
            .frame(maxHeight: 200)
            .background(.white, in: RoundedRectangle(cornerRadius: 18))
        }
        .padding(.horizontal, 20).padding(.bottom, 14)
        .transition(.move(edge: .bottom).combined(with: .opacity))
    }

    private func cameraMessage(icon: String, title: String, message: String) -> some View {
        VStack(spacing: 12) {
            Image(systemName: icon).font(.system(size: 36))
            Text(title).font(.headline)
            Text(message).font(.subheadline).foregroundStyle(.secondary).multilineTextAlignment(.center)
        }
        .padding(24).background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 20)).padding(32)
    }
}

private enum DownloadSizeUnit {
    case kilobytes
    case megabytes
    case gigabytes
    case terabytes

    static func forTotalBytes(_ bytes: Int64) -> Self {
        switch bytes {
        case 1_000_000_000_000...: .terabytes
        case 1_000_000_000...: .gigabytes
        case 1_000_000...: .megabytes
        default: .kilobytes
        }
    }

    func format(_ bytes: Int64) -> String {
        let divisor: Double
        let suffix: String
        switch self {
        case .kilobytes:
            divisor = 1_000
            suffix = "KB"
        case .megabytes:
            divisor = 1_000_000
            suffix = "MB"
        case .gigabytes:
            divisor = 1_000_000_000
            suffix = "GB"
        case .terabytes:
            divisor = 1_000_000_000_000
            suffix = "TB"
        }
        return String(format: "%.2f %@", Double(bytes) / divisor, suffix)
    }
}

#Preview { ContentView() }
