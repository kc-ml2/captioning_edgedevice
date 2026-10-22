#if INTERNAL_TESTING
import Foundation
import UIKit

nonisolated struct InternalReviewError: LocalizedError {
    let status: Int
    var permanent: Bool { [400, 401, 403, 409, 413, 422].contains(status) }
    var errorDescription: String? {
        switch status {
        case 401, 403: String(localized: "Review authentication failed. Contact the test administrator.")
        case 409: String(localized: "Feedback already exists for this caption. Close this screen; another submission is not allowed.")
        case 400, 413, 422: String(localized: "The review was rejected. Close this screen and check the comment or photo before trying again.")
        default: String(localized: "The server is unavailable. Try sending again without closing this screen.")
        }
    }
}

actor InternalReviewClient {
    func send(_ body: Data) async throws {
                guard let key = TelemetryConfiguration.ingestAPIKey else {
                    throw CocoaError(.fileReadNoPermission)
                }
                var request = URLRequest(url: TelemetryConfiguration.baseURL.appending(path: "v1/internal/reviews"))
                request.httpMethod = "POST"
                request.httpBody = body
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                request.setValue(key, forHTTPHeaderField: "X-Internal-Review-Key")
                request.timeoutInterval = 60
                let configuration = URLSessionConfiguration.ephemeral
                let session = URLSession(configuration: configuration)
                defer { session.finishTasksAndInvalidate() }
                let (_, response) = try await session.data(for: request)
                guard let http = response as? HTTPURLResponse else { throw CocoaError(.fileWriteUnknown) }
                if http.statusCode != 200 { throw InternalReviewError(status: http.statusCode) }
    }
}

nonisolated struct InternalReviewRequest: Encodable, Sendable {
    let review_id: UUID
    let benchmark_event_id: UUID
    let caption: String
    let comment: String
    let model_version: String
    var image_base64: String?
}

nonisolated func prepareReview(_ input: InternalReviewRequest, image: CGImage?) throws -> Data {
    var request = input
    if let image {
        let size = CGSize(width: image.width, height: image.height)
        let scale = min(1, 1600 / max(size.width, size.height))
        let target = CGSize(width: size.width * scale, height: size.height * scale)
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        let resized = UIGraphicsImageRenderer(size: target, format: format).image { _ in
            UIImage(cgImage: image).draw(in: CGRect(origin: .zero, size: target))
        }
        guard let jpeg = resized.jpegData(compressionQuality: 0.8) else { throw CocoaError(.fileWriteUnknown) }
        request.image_base64 = jpeg.base64EncodedString()
    }
    return try JSONEncoder().encode(request)
}
#endif
