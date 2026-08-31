import AVFoundation
import Combine
import SwiftUI
import UIKit

@MainActor
final class CameraModel: ObservableObject {
    @Published private(set) var permissionDenied = false
    @Published private(set) var isUnavailable = false

    let session = AVCaptureSession()
    private let photoOutput = AVCapturePhotoOutput()
    private var photoDelegate: PhotoCaptureDelegate?
    private var isConfigured = false

    func start() async {
        let authorized: Bool

        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            authorized = true
        case .notDetermined:
            authorized = await AVCaptureDevice.requestAccess(for: .video)
        default:
            authorized = false
        }

        guard authorized else {
            permissionDenied = true
            return
        }

        configureIfNeeded()
        guard isConfigured else { return }

        if !session.isRunning {
            session.startRunning()
        }
    }

    func stop() {
        if session.isRunning {
            session.stopRunning()
        }
    }

    func captureFrame() async throws -> CGImage {
        #if targetEnvironment(simulator)
        if let url = Bundle.main.url(forResource: "DebugSample", withExtension: "jpg"),
           let image = UIImage(contentsOfFile: url.path)?.cgImage {
            return image
        }
        #endif

        guard isConfigured, session.isRunning else {
            throw CameraError.notReady
        }

        return try await withCheckedThrowingContinuation { continuation in
            let delegate = PhotoCaptureDelegate { [weak self] result in
                self?.photoDelegate = nil
                continuation.resume(with: result)
            }
            photoDelegate = delegate
            photoOutput.capturePhoto(with: AVCapturePhotoSettings(), delegate: delegate)
        }
    }

    private func configureIfNeeded() {
        guard !isConfigured else { return }
        guard let camera = AVCaptureDevice.default(
            .builtInWideAngleCamera,
            for: .video,
            position: .back
        ) else {
            isUnavailable = true
            return
        }

        do {
            let input = try AVCaptureDeviceInput(device: camera)
            session.beginConfiguration()
            session.sessionPreset = .photo

            guard session.canAddInput(input) else {
                session.commitConfiguration()
                isUnavailable = true
                return
            }

            session.addInput(input)
            guard session.canAddOutput(photoOutput) else {
                session.commitConfiguration()
                isUnavailable = true
                return
            }
            session.addOutput(photoOutput)
            session.commitConfiguration()
            isConfigured = true
        } catch {
            isUnavailable = true
        }
    }
}

private enum CameraError: LocalizedError {
    case notReady
    case captureFailed

    var errorDescription: String? {
        switch self {
        case .notReady: L10n.cameraNotReady
        case .captureFailed: L10n.photoCaptureFailed
        }
    }
}

private final class PhotoCaptureDelegate: NSObject, AVCapturePhotoCaptureDelegate {
    private let completion: (Result<CGImage, Error>) -> Void

    init(completion: @escaping (Result<CGImage, Error>) -> Void) {
        self.completion = completion
    }

    func photoOutput(
        _ output: AVCapturePhotoOutput,
        didFinishProcessingPhoto photo: AVCapturePhoto,
        error: Error?
    ) {
        if let error {
            completion(.failure(error))
            return
        }
        guard let data = photo.fileDataRepresentation(),
              let image = UIImage(data: data)?.cgImage else {
            completion(.failure(CameraError.captureFailed))
            return
        }
        completion(.success(image))
    }
}

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {
        uiView.previewLayer.session = session
    }
}

final class PreviewView: UIView {
    override class var layerClass: AnyClass {
        AVCaptureVideoPreviewLayer.self
    }

    var previewLayer: AVCaptureVideoPreviewLayer {
        layer as! AVCaptureVideoPreviewLayer
    }
}
