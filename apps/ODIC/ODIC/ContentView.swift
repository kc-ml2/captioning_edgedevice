import SwiftUI

struct ContentView: View {
    @Environment(\.scenePhase) private var scenePhase
    @StateObject private var camera = CameraModel()
    @State private var isCheckingModel = true
    @State private var isCaptioning = false
    @State private var isInstallingModel = false
    @State private var modelReady = false
    @State private var modelProgress = 0.0
    @State private var downloadedBytes: Int64 = 0
    @State private var totalDownloadBytes: Int64 = 0
    @State private var modelDownloadSize: String?
    @State private var modelStatus: String?
    @State private var caption: String?
    @State private var errorMessage: String?
    @State private var installationTask: Task<Void, Never>?

    private let pipeline = CaptionPipeline()
    private let modelStore = ModelStore()

    var body: some View {
        Group {
            if modelReady {
                cameraScreen
            } else {
                modelSetupScreen
            }
        }
        .task { await prepareApp() }
        .onChange(of: scenePhase) { _, phase in
            guard modelReady else { return }
            if phase == .active { Task { await camera.start() } }
            if phase == .background { camera.stop() }
        }
        .preferredColorScheme(.dark)
    }

    private var cameraScreen: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            CameraPreview(session: camera.session).ignoresSafeArea()

            if camera.permissionDenied {
                cameraMessage(icon: "camera.fill", title: "카메라 접근이 필요해요", message: "설정에서 ODIC의 카메라 접근을 허용해 주세요.")
            } else if camera.isUnavailable {
                #if targetEnvironment(simulator)
                cameraMessage(icon: "photo.fill", title: "샘플 이미지 모드", message: "말풍선 버튼을 누르면 샘플 이미지로 캡션을 생성합니다.")
                #else
                cameraMessage(icon: "camera.slash.fill", title: "카메라를 사용할 수 없어요", message: "실제 iPhone에서 다시 시도해 주세요.")
                #endif
            }

            VStack(spacing: 0) {
                topBar
                Spacer()
                if let caption { resultBubble(caption) }
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
                    Image(systemName: isInstallingModel ? "arrow.down.circle.fill" : "shippingbox.fill")
                        .font(.system(size: 58, weight: .medium))
                        .foregroundStyle(.orange)
                        .symbolEffect(.pulse, isActive: isInstallingModel)

                    VStack(spacing: 8) {
                        Text(isCheckingModel ? "모델 확인 중" : "온디바이스 모델 준비")
                            .font(.title2.bold())
                        Text(setupDescription)
                            .font(.subheadline)
                            .foregroundStyle(.white.opacity(0.7))
                            .multilineTextAlignment(.center)
                            .lineSpacing(3)
                    }

                    if isInstallingModel {
                        VStack(spacing: 10) {
                            ProgressView(value: modelProgress)
                                .tint(.orange)
                            HStack {
                                Text(downloadProgressText)
                                Spacer()
                                Text(modelProgress, format: .percent.precision(.fractionLength(0)))
                            }
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.white.opacity(0.65))
                        }
                    }

                    if let modelStatus {
                        Text(modelStatus)
                            .font(.footnote.weight(.medium))
                            .foregroundStyle(.white.opacity(0.8))
                            .multilineTextAlignment(.center)
                    }

                    if let errorMessage {
                        Label(errorMessage, systemImage: "exclamationmark.triangle.fill")
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

                Label("다운로드 중에는 카메라를 사용하지 않아요", systemImage: "camera.fill")
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.55))
                    .padding(.bottom, 24)
            }
        }
    }

    @ViewBuilder
    private var setupButton: some View {
        if isInstallingModel {
            Button(role: .cancel) { cancelInstallation() } label: {
                Label("다운로드 취소", systemImage: "xmark.circle.fill")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .tint(.white)
        } else {
            Button { modelDownloadSize == nil ? retryModelCheck() : installModel() } label: {
                Label(
                    modelDownloadSize == nil ? "다시 확인" : "모델 다운로드",
                    systemImage: modelDownloadSize == nil ? "arrow.clockwise" : "icloud.and.arrow.down.fill"
                )
                .fontWeight(.semibold)
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .tint(.orange)
            .disabled(isCheckingModel)
        }
    }

    private var setupDescription: String {
        if isCheckingModel { return "설치된 모델과 최신 버전을 확인하고 있어요." }
        if let modelDownloadSize {
            return "이미지 설명은 기기 안에서 처리됩니다.\n한 번만 모델을 내려받아 주세요 (\(modelDownloadSize))."
        }
        return "이미지 설명에 필요한 모델 정보를 불러오지 못했어요."
    }

    private var downloadProgressText: String {
        let unit = DownloadSizeUnit.forTotalBytes(totalDownloadBytes)
        let completed = unit.format(downloadedBytes)
        let total = unit.format(totalDownloadBytes)
        return "\(completed) / \(total)"
    }

    private func prepareApp() async {
        modelReady = await modelStore.isInstalled
        if modelReady {
            isCheckingModel = false
            await camera.start()
            return
        }

        camera.stop()
        do {
            let availability = try await modelStore.latestAvailability()
            totalDownloadBytes = availability.totalSize
            modelDownloadSize = availability.formattedSize
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
        }
        isCheckingModel = false
    }

    private func retryModelCheck() {
        isCheckingModel = true
        errorMessage = nil
        Task {
            do {
                let availability = try await modelStore.latestAvailability()
                totalDownloadBytes = availability.totalSize
                modelDownloadSize = availability.formattedSize
            } catch {
                errorMessage = error.localizedDescription
            }
            isCheckingModel = false
        }
    }

    private func installModel() {
        installationTask?.cancel()
        camera.stop()
        isInstallingModel = true
        errorMessage = nil
        modelStatus = "모델 다운로드를 준비하고 있어요…"
        modelProgress = 0
        downloadedBytes = 0

        installationTask = Task {
            do {
                try await modelStore.installLatest { progress in
                    await MainActor.run {
                        modelStatus = progress.message
                        modelProgress = max(modelProgress, progress.fraction)
                        downloadedBytes = max(downloadedBytes, progress.completedBytes)
                        if progress.totalBytes > 0 {
                            totalDownloadBytes = progress.totalBytes
                        }
                    }
                }
                try Task.checkCancellation()
                modelReady = true
                modelStatus = nil
                isInstallingModel = false
                installationTask = nil
                if scenePhase == .active { await camera.start() }
            } catch is CancellationError {
                modelStatus = nil
                errorMessage = "다운로드를 취소했어요."
                isInstallingModel = false
                installationTask = nil
            } catch {
                modelStatus = nil
                errorMessage = error.localizedDescription
                isInstallingModel = false
                installationTask = nil
            }
        }
    }

    private func cancelInstallation() {
        modelStatus = "다운로드를 취소하고 있어요…"
        installationTask?.cancel()
    }

    private var topBar: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("ODIC").font(.headline.weight(.bold))
                Text("On-Device Image Captioning").font(.caption2).foregroundStyle(.white.opacity(0.7))
            }
            Spacer()
            Image(systemName: "lock.shield.fill")
                .font(.system(size: 16, weight: .semibold))
                .padding(10).background(.ultraThinMaterial, in: Circle())
                .accessibilityLabel("온디바이스 처리")
        }
        .padding(.horizontal, 20).padding(.top, 8).padding(.bottom, 18)
        .background(LinearGradient(colors: [.black.opacity(0.75), .clear], startPoint: .top, endPoint: .bottom).ignoresSafeArea(edges: .top))
    }

    private var captionPanel: some View {
        VStack(spacing: 16) {
            Text(captionStatusText)
                .font(.subheadline.weight(.medium))
                .foregroundStyle(errorMessage == nil ? .white.opacity(0.9) : .orange)
                .multilineTextAlignment(.center)
                .padding(.horizontal)

            Button { generateCaption() } label: {
                ZStack {
                    Circle().stroke(.white.opacity(0.95), lineWidth: 4).frame(width: 82, height: 82)
                    Circle().fill(isCaptioning ? Color.orange : Color.white).frame(width: 66, height: 66)
                    Image(systemName: isCaptioning ? "ellipsis.message.fill" : "text.bubble.fill")
                        .font(.system(size: 29, weight: .semibold)).foregroundStyle(.black)
                        .symbolEffect(.pulse, isActive: isCaptioning)
                }
            }
            .buttonStyle(.plain).disabled(isCaptioning)
            .accessibilityLabel("이미지 캡션 생성")
        }
        .frame(maxWidth: .infinity).padding(.top, 18).padding(.bottom, 18)
        .background(.black.opacity(0.72))
    }

    private var captionStatusText: String {
        if isCaptioning { return "장면을 설명하고 있어요…" }
        if let errorMessage { return errorMessage }
        return "버튼을 눌러 장면을 설명해 보세요"
    }

    private func resultBubble(_ text: String) -> some View {
        Text(text)
            .font(.body.weight(.medium)).foregroundStyle(.black)
            .padding(.horizontal, 18).padding(.vertical, 14)
            .background(.white, in: RoundedRectangle(cornerRadius: 18))
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 20).padding(.bottom, 14)
            .transition(.move(edge: .bottom).combined(with: .opacity))
    }

    private func generateCaption() {
        isCaptioning = true
        caption = nil
        errorMessage = nil

        Task {
            do {
                let frame = try await camera.captureFrame()
                let generated = try await pipeline.caption(image: frame) { status in
                    await MainActor.run { errorMessage = status }
                }
                withAnimation {
                    caption = generated
                    errorMessage = nil
                }
            } catch {
                errorMessage = error.localizedDescription
            }
            isCaptioning = false
        }
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
