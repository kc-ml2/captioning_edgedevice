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
    @State private var failureMessage = ""
    @State private var permanentFailure = false
    @State private var draft = InternalReviewDraft()

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Text(caption)
                    ZStack(alignment: .topLeading) {
                        TextEditor(text: $comment).frame(minHeight: 120)
                            .accessibilityLabel("Review comment")
                            .disabled(isSending || draft.isFrozen)
                        if comment.isEmpty {
                            Text("Enter your comment here.")
                                .foregroundStyle(.secondary)
                                .padding(.horizontal, 5).padding(.vertical, 8)
                                .allowsHitTesting(false)
                                .accessibilityHidden(true)
                        }
                    }
                    Toggle("Attach the caption source photo", isOn: $attachImage)
                        .disabled(image == nil || isSending || draft.isFrozen)
                }
                Section {
                    Button("Share with team") { send() }
                        .disabled(isSending || comment.unicodeScalars.count > 5000 || permanentFailure)
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
                Text(failureMessage)
            }
        }
    }

    private func send() {
        isSending = true
        Task {
            defer { isSending = false }
            do {
                if !draft.isFrozen {
                    let request = InternalReviewRequest(review_id: draft.id, benchmark_event_id: benchmark.eventID,
                        caption: caption, comment: comment, model_version: benchmark.modelVersion, image_base64: nil)
                    let source = attachImage ? image : nil
                    draft.body = try await Task.detached(priority: .userInitiated) {
                        try prepareReview(request, image: source)
                    }.value
                }
                guard let pendingBody = draft.body else { return }
                try await InternalReviewClient().send(pendingBody)
                onSubmitted()
                dismiss()
            } catch let error as InternalReviewError {
                permanentFailure = error.permanent
                failureMessage = error.localizedDescription
                failed = true
            } catch {
                failureMessage = String(localized: "Try sending again. Closing this screen discards the draft; a previous request may already have reached the server.")
                failed = true
            }
        }
    }
}

#endif
