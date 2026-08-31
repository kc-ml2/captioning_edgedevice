import Foundation

enum L10n {
    // Camera UI
    static let cameraPermissionTitle = String(localized: "camera.permission.title", defaultValue: "Camera Access Required")
    static let cameraPermissionMessage = String(localized: "camera.permission.message", defaultValue: "Allow ODIC to access the camera in Settings.")
    static let sampleModeTitle = String(localized: "camera.sample_mode.title", defaultValue: "Sample Image Mode")
    static let sampleModeMessage = String(localized: "camera.sample_mode.message", defaultValue: "Tap the speech bubble button to generate a caption from the sample image.")
    static let cameraUnavailableTitle = String(localized: "camera.unavailable.title", defaultValue: "Camera Unavailable")
    static let cameraUnavailableMessage = String(localized: "camera.unavailable.message", defaultValue: "Try again on a physical iPhone.")
    static let cameraNotReady = String(localized: "camera.error.not_ready", defaultValue: "The camera is not ready yet.")
    static let photoCaptureFailed = String(localized: "camera.error.capture_failed", defaultValue: "Could not capture the photo.")

    // Model setup UI
    static let checkingModelTitle = String(localized: "model.setup.checking.title", defaultValue: "Checking Model")
    static let prepareModelTitle = String(localized: "model.setup.prepare.title", defaultValue: "Prepare On-Device Model")
    static let noCameraDuringDownload = String(localized: "model.setup.no_camera_during_download", defaultValue: "The camera is not used while downloading")
    static let cancelDownload = String(localized: "model.setup.cancel_download", defaultValue: "Cancel Download")
    static let retry = String(localized: "model.setup.retry", defaultValue: "Try Again")
    static let downloadModel = String(localized: "model.setup.download", defaultValue: "Download Model")
    static let checkingInstalledModel = String(localized: "model.setup.checking.description", defaultValue: "Checking the installed model and latest version.")
    static let modelInfoUnavailable = String(localized: "model.setup.info_unavailable", defaultValue: "Could not load the model information required for image descriptions.")
    static let preparingDownload = String(localized: "model.download.preparing", defaultValue: "Preparing the model download…")
    static let downloadCancelled = String(localized: "model.download.cancelled", defaultValue: "The download was cancelled.")
    static let cancellingDownload = String(localized: "model.download.cancelling", defaultValue: "Cancelling the download…")

    static func modelDownloadDescription(size: String) -> String {
        String(
            format: String(localized: "model.setup.download.description", defaultValue: "Image descriptions are processed on your device.\nDownload the model once (%@)."),
            locale: .current,
            size
        )
    }

    // Caption UI (the generated caption itself remains unchanged)
    static let onDeviceProcessing = String(localized: "caption.on_device_processing", defaultValue: "On-device processing")
    static let generateCaption = String(localized: "caption.generate", defaultValue: "Generate Image Caption")
    static let describingScene = String(localized: "caption.status.describing", defaultValue: "Describing the scene…")
    static let captionPrompt = String(localized: "caption.status.prompt", defaultValue: "Tap the button to describe the scene")
    static let inspectingImage = String(localized: "caption.progress.inspecting_image", defaultValue: "Inspecting the image…")
    static let understandingScene = String(localized: "caption.progress.understanding_scene", defaultValue: "Understanding the scene…")
    static let creatingDescription = String(localized: "caption.progress.creating_description", defaultValue: "Creating a description…")
    static let loadingModel = String(localized: "caption.progress.loading_model", defaultValue: "Loading the model…")

    // Pipeline errors
    static let modelNotInstalled = String(localized: "pipeline.error.model_not_installed", defaultValue: "Could not find the MobileVLM model.")
    static let invalidCameraFrame = String(localized: "pipeline.error.invalid_frame", defaultValue: "Could not read the camera frame.")
    static let fourBitModelRequired = String(localized: "pipeline.error.four_bit_required", defaultValue: "An MLX 4-bit model is required on iPhone.")

    static func missingWeight(_ name: String) -> String {
        String(format: String(localized: "pipeline.error.missing_weight", defaultValue: "Missing model weight: %@"), locale: .current, name)
    }

    static func unsupportedModel(_ reason: String) -> String {
        String(format: String(localized: "pipeline.error.unsupported_model", defaultValue: "Unsupported model: %@"), locale: .current, reason)
    }

    // Model store
    static let endpointMissing = String(localized: "model.error.endpoint_missing", defaultValue: "The model download URL is not configured.")
    static let invalidServerResponse = String(localized: "model.error.invalid_response", defaultValue: "The model server returned an invalid response.")
    static let unsupportedManifest = String(localized: "model.error.unsupported_manifest", defaultValue: "Unsupported model manifest.")
    static let checkingLatestModel = String(localized: "model.progress.checking_latest", defaultValue: "Checking the latest model…")
    static let modelReady = String(localized: "model.progress.ready", defaultValue: "The model is ready.")
    static let installationComplete = String(localized: "model.progress.installation_complete", defaultValue: "Model installation is complete.")

    static func unsafePath(_ path: String) -> String {
        String(format: String(localized: "model.error.unsafe_path", defaultValue: "Unsafe model file path: %@"), locale: .current, path)
    }

    static func invalidFile(_ path: String) -> String {
        String(format: String(localized: "model.error.invalid_file", defaultValue: "The downloaded model file is damaged: %@"), locale: .current, path)
    }

    static func insufficientStorage(required: String, available: String) -> String {
        String(
            format: String(localized: "model.error.insufficient_storage", defaultValue: "Not enough storage. Required: %@, available: %@"),
            locale: .current,
            required,
            available
        )
    }

    static func downloadingFile(_ filename: String) -> String {
        String(format: String(localized: "model.progress.downloading_file", defaultValue: "Downloading model: %@"), locale: .current, filename)
    }

    static func verifyingFile(_ filename: String) -> String {
        String(format: String(localized: "model.progress.verifying_file", defaultValue: "Verifying file: %@"), locale: .current, filename)
    }
}
