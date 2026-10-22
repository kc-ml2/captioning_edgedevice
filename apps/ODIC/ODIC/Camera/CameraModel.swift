import AVFoundation
import Combine
import SwiftUI
import UIKit

@MainActor
final class CameraModel: ObservableObject {
    @Published private(set) var permissionDenied = false
    @Published private(set) var isUnavailable = false

    private let capture = CameraCapture()
    var session: AVCaptureSession { capture.session }

    func start() async {
        let authorized: Bool
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: authorized = true
        case .notDetermined: authorized = await AVCaptureDevice.requestAccess(for: .video)
        default: authorized = false
        }
        permissionDenied = !authorized
        guard authorized else { return }
        isUnavailable = !(await capture.start())
    }

    func stop() { capture.stop() }

    func captureFrame() async throws -> CGImage {
        #if targetEnvironment(simulator)
        if let url = Bundle.main.url(forResource: "DebugSample", withExtension: "jpg"),
           let image = UIImage(contentsOfFile: url.path)?.cgImage { return image }
        #endif
        return try await capture.frame()
    }
}

// All mutable capture state is confined to queue. The preview only connects session.
nonisolated private final class CameraCapture: @unchecked Sendable {
    let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "scenesense.camera")
    private let output = AVCapturePhotoOutput()
    private var configured = false
    private var delegate: PhotoCaptureDelegate?

    func start() async -> Bool {
        await withCheckedContinuation { continuation in
            queue.async {
                do {
                    if !self.configured {
                        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)
                        else { continuation.resume(returning: false); return }
                        let input = try AVCaptureDeviceInput(device: device)
                        self.session.beginConfiguration()
                        self.session.automaticallyConfiguresApplicationAudioSession = false
                        self.session.sessionPreset = .photo
                        guard self.session.canAddInput(input), self.session.canAddOutput(self.output) else {
                            self.session.commitConfiguration()
                            continuation.resume(returning: false)
                            return
                        }
                        self.session.addInput(input)
                        self.session.addOutput(self.output)
                        self.session.commitConfiguration()
                        self.configured = true
                    }
                    if !self.session.isRunning { self.session.startRunning() }
                    continuation.resume(returning: true)
                } catch { continuation.resume(returning: false) }
            }
        }
    }

    func stop() {
        queue.async {
            if self.session.isRunning { self.session.stopRunning() }
            self.delegate?.finish(.failure(CancellationError()))
            self.delegate = nil
        }
    }

    func frame() async throws -> CGImage {
        try Task.checkCancellation()
        return try await withCheckedThrowingContinuation { continuation in
            queue.async {
                guard self.session.isRunning, self.delegate == nil else {
                    continuation.resume(throwing: CameraError.notReady)
                    return
                }
                let delegate = PhotoCaptureDelegate { result in
                    self.queue.async { self.delegate = nil; continuation.resume(with: result) }
                }
                self.delegate = delegate
                self.output.capturePhoto(with: AVCapturePhotoSettings(), delegate: delegate)
            }
        }
    }
}

nonisolated private enum CameraError: LocalizedError {
    case notReady
    case captureFailed

    var errorDescription: String? {
        switch self {
        case .notReady: L10n.cameraNotReady
        case .captureFailed: L10n.photoCaptureFailed
        }
    }
}

nonisolated private final class PhotoCaptureDelegate: NSObject, AVCapturePhotoCaptureDelegate, @unchecked Sendable {
    private let lock = NSLock()
    private var completed = false
    private let completion: @Sendable (Result<CGImage, Error>) -> Void

    init(completion: @escaping @Sendable (Result<CGImage, Error>) -> Void) {
        self.completion = completion
    }

    func finish(_ result: Result<CGImage, Error>) {
        lock.lock()
        guard !completed else { lock.unlock(); return }
        completed = true
        lock.unlock()
        completion(result)
    }

    func photoOutput(_ output: AVCapturePhotoOutput, didFinishCaptureFor resolvedSettings: AVCaptureResolvedPhotoSettings, error: Error?) {
        if let error { finish(.failure(error)) }
    }

    func photoOutput(
        _ output: AVCapturePhotoOutput,
        didFinishProcessingPhoto photo: AVCapturePhoto,
        error: Error?
    ) {
        if let error {
            finish(.failure(error))
            return
        }
        guard let data = photo.fileDataRepresentation(),
              let source = UIImage(data: data) else {
            finish(.failure(CameraError.captureFailed))
            return
        }
        // Bake EXIF orientation into pixels before discarding UIImage metadata.
        // The same upright frame is used for inference and internal attachments.
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        let normalized = UIGraphicsImageRenderer(size: source.size, format: format).image { _ in
            source.draw(in: CGRect(origin: .zero, size: source.size))
        }
        guard let image = normalized.cgImage else {
            finish(.failure(CameraError.captureFailed))
            return
        }
        finish(.success(image))
    }
}
