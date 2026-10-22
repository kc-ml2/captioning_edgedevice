import SwiftUI
import Combine

@MainActor
final class CaptionScreenModel: ObservableObject {
    let camera = CameraModel()
    let speech = CaptionSpeech()
    private var scenePhase: ScenePhase = .active
    private var observations: Set<AnyCancellable> = []

    init() {
        camera.objectWillChange.sink { [weak self] _ in self?.objectWillChange.send() }.store(in: &observations)
        speech.objectWillChange.sink { [weak self] _ in self?.objectWillChange.send() }.store(in: &observations)
    }
    @Published var isCheckingModel = true
    @Published var isCaptioning = false
    @Published var isInstallingModel = false
    @Published var modelReady = false
    @Published var modelProgress = 0.0
    @Published var downloadedBytes: Int64 = 0
    @Published var totalDownloadBytes: Int64 = 0
    @Published var modelDownloadSize: String?
    @Published var modelStatus: String?
    @Published var caption: String?
    @Published var captionID = UUID()
    @Published var captionBenchmark: BenchmarkReference?
    @Published var captionSourceImage: CGImage?
    @Published var errorMessage: String?
    private var installationTask: Task<Void, Never>?

    private var generationTask: Task<Void, Never>?
    private var generationID = UUID()
    private var pipeline = CaptionPipeline()
    private var needsModelRepair = false
    private let modelStore = ModelStore()

    func prepareApp() async {
        modelReady = await modelStore.isInstalled
        if modelReady {
            isCheckingModel = false
            if scenePhase == .active { await camera.start() }
            if scenePhase != .active { camera.stop() }
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

    func retryModelCheck() {
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

    func installModel() {
        installationTask?.cancel()
        camera.stop()
        isInstallingModel = true
        errorMessage = nil
        modelStatus = L10n.preparingDownload
        modelProgress = 0
        downloadedBytes = 0

        installationTask = Task {
            let startedAt = ProcessInfo.processInfo.systemUptime
            do {
                if needsModelRepair { try? await modelStore.invalidateInstallation() }
                try await modelStore.installLatest { [self] progress in
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
                needsModelRepair = false
                pipeline = CaptionPipeline()
                modelStatus = nil
                isInstallingModel = false
                installationTask = nil
                if scenePhase == .active { await camera.start() }
                if scenePhase != .active { camera.stop() }
            } catch is CancellationError {
                modelStatus = nil
                errorMessage = L10n.downloadCancelled
                isInstallingModel = false
                installationTask = nil
            } catch {
                if !Task.isCancelled {
                    await BenchmarkTelemetry.recordFailure(error, kind: "download_failure",
                        startedAt: startedAt, appState: telemetryAppState,
                        downloadedBytes: downloadedBytes)
                }
                modelStatus = nil
                errorMessage = error.localizedDescription
                isInstallingModel = false
                installationTask = nil
            }
        }
    }

    func cancelInstallation() {
        modelStatus = L10n.cancellingDownload
        installationTask?.cancel()
    }

    private var telemetryAppState: String {
        switch scenePhase {
        case .active: "active"
        case .inactive: "inactive"
        case .background: "background"
        @unknown default: "unknown"
        }
    }

    func generateCaption() {
        speech.stop()
        isCaptioning = true
        caption = nil
        captionID = UUID()
        captionBenchmark = nil
        captionSourceImage = nil
        errorMessage = nil

        generationTask?.cancel()
        let requestID = UUID()
        generationID = requestID
        generationTask = Task {
            let startedAt = ProcessInfo.processInfo.systemUptime
            var stage = "capture"
            do {
                let frame = try await camera.captureFrame()
                try Task.checkCancellation()
                stage = "inference"
                speech.beginCaption()
                let generated = try await pipeline.caption(image: frame, progress: { [self] status in
                    await MainActor.run { if generationID == requestID { errorMessage = status } }
                }, partialCaption: { [self] partial in
                    await MainActor.run {
                        if generationID == requestID, !partial.isEmpty {
                            caption = partial
                            speech.appendStreaming(partial, final: false)
                        }
                    }
                })
                try Task.checkCancellation()
                guard generationID == requestID else { return }
                withAnimation {
                    #if INTERNAL_TESTING
                    captionSourceImage = frame
                    #endif
                    captionBenchmark = generated.benchmark
                    caption = generated.text
                    errorMessage = nil
                }
            } catch is CancellationError {
                // Keep any partial text so suspended speech can resume on return.
                errorMessage = nil
            } catch {
                guard generationID == requestID else { return }
                caption = nil
                if stage == "inference" {
                    needsModelRepair = true
                    modelReady = false
                    camera.stop()
                    if let availability = try? await modelStore.latestAvailability() {
                        totalDownloadBytes = availability.totalSize
                        modelDownloadSize = availability.formattedSize
                    }
                }
                await BenchmarkTelemetry.recordFailure(error, kind: "caption_failure",
                    startedAt: startedAt, appState: telemetryAppState, stage: stage)
                errorMessage = error.localizedDescription
            }
            if generationID == requestID {
                isCaptioning = false
                if let caption { speech.appendStreaming(caption, final: true) }
                generationTask = nil
            }
        }
    }

    func sceneChanged(_ phase: ScenePhase) {
        scenePhase = phase
            if phase == .active { speech.resume() }
            else { speech.suspend() }
            if phase == .background { generationTask?.cancel(); camera.stop() }
            guard modelReady else { return }
            if phase == .active {
                Task { await camera.start(); if scenePhase != .active { camera.stop() } }
            }
    }
}
