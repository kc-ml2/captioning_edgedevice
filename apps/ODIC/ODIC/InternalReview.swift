#if INTERNAL_TESTING
import Foundation
import SwiftUI
import UIKit

struct InternalReviewForm: View {
    let caption: String
    let image: CGImage?
    let benchmark: BenchmarkReference
    let onSubmitted: () -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var comment = ""
    @State private var attachImage = false
    @State private var isSending = false
    @State private var failed = false
    @State private var reviewID = UUID()
    // Freeze retries to the same ID/body. Editing after a failed attempt creates a new review.
    @State private var pendingBody: Data?

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Text(caption)
                    ZStack(alignment: .topLeading) {
                        TextEditor(text: $comment).frame(minHeight: 120)
                            .accessibilityLabel("Review comment")
                            .disabled(isSending || pendingBody != nil)
                        if comment.isEmpty {
                            Text("Enter your comment here.")
                                .foregroundStyle(.secondary)
                                .padding(.horizontal, 5).padding(.vertical, 8)
                                .allowsHitTesting(false)
                                .accessibilityHidden(true)
                        }
                    }
                    Toggle("Attach the caption source photo", isOn: $attachImage)
                        .disabled(image == nil || isSending || pendingBody != nil)
                }
                Section {
                    Button("Share with team") { send() }
                        .disabled(isSending || comment.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || comment.count > 5000)
                    if isSending { ProgressView() }
                } footer: {
                    Text("Share this result and your comment with the team. This is not a caption rating. The photo is sent only when attached.")
                }
            }
            .navigationTitle("Share with team")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }.disabled(isSending)
                }
            }
            .interactiveDismissDisabled(isSending)
            .alert("Review could not be sent", isPresented: $failed) {
                Button("OK", role: .cancel) { }
            } message: {
                Text("Try sending again. Closing this screen discards the draft; a previous request may already have reached the server.")
            }
        }
    }

    private func send() {
        isSending = true
        Task {
            defer { isSending = false }
            do {
                if pendingBody == nil {
                    var photo: String?
                    if attachImage, let image {
                        let size = CGSize(width: image.width, height: image.height)
                        let scale = min(1, 1600 / max(size.width, size.height))
                        let target = CGSize(width: size.width * scale, height: size.height * scale)
                        let format = UIGraphicsImageRendererFormat()
                        format.scale = 1
                        let resized = UIGraphicsImageRenderer(size: target, format: format).image { _ in
                            UIImage(cgImage: image).draw(in: CGRect(origin: .zero, size: target))
                        }
                        guard let jpeg = resized.jpegData(compressionQuality: 0.8) else {
                            throw CocoaError(.fileWriteUnknown)
                        }
                        photo = jpeg.base64EncodedString()
                    }
                    pendingBody = try JSONEncoder().encode(InternalReviewRequest(
                        review_id: reviewID, benchmark_event_id: benchmark.eventID, caption: caption,
                        comment: comment, model_version: benchmark.modelVersion, image_base64: photo
                    ))
                }
                guard let key = TelemetryConfiguration.ingestAPIKey else {
                    throw CocoaError(.fileReadNoPermission)
                }
                var request = URLRequest(url: TelemetryConfiguration.baseURL.appending(path: "v1/internal/reviews"))
                request.httpMethod = "POST"
                request.httpBody = pendingBody
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                request.setValue(key, forHTTPHeaderField: "X-Internal-Review-Key")
                request.timeoutInterval = 60
                let configuration = URLSessionConfiguration.ephemeral
                let session = URLSession(configuration: configuration)
                defer { session.finishTasksAndInvalidate() }
                let (_, response) = try await session.data(for: request)
                guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                    throw CocoaError(.fileWriteUnknown)
                }
                onSubmitted()
                dismiss()
            } catch { failed = true }
        }
    }
}

nonisolated private struct InternalReviewRequest: Encodable {
    let review_id: UUID
    let benchmark_event_id: UUID
    let caption: String
    let comment: String
    let model_version: String
    let image_base64: String?
}
#endif
