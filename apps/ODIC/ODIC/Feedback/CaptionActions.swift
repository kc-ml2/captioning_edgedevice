import AVFoundation
import Combine
import OSLog
import SwiftUI

/// One instance per displayed caption. Never persists the caption or image.
struct CaptionActions: View {
    let caption: String
    let benchmark: BenchmarkReference?
    var sourceImage: CGImage? = nil
    #if INTERNAL_TESTING
    @State private var showingInternalReview = false
    #endif
    @ObservedObject var speech: CaptionSpeech
    @State private var selection: Set<FeedbackChoice> = []
    @State private var submittedSelection: Set<FeedbackChoice> = []
    @State private var internalFeedbackSubmitted = false
    @State private var isSaving = false
    @State private var showingFeedback = false
    @State private var saveError = false

    var body: some View {
        HStack {
            Button {
                speech.stop()
                selection = submittedSelection
                showingFeedback = true
            } label: {
                Label("Feedback", systemImage: submittedSelection.isEmpty ? "bubble.left" : "checkmark.bubble")
            }
            .disabled(benchmark == nil || !submittedSelection.isEmpty)
            #if INTERNAL_TESTING
            Button {
                speech.stop()
                showingInternalReview = true
            } label: {
                Label("Share with team", systemImage: "square.and.arrow.up")
            }
            .disabled(benchmark == nil || internalFeedbackSubmitted)
            #endif
        }
        .buttonStyle(.bordered)
        .tint(.white)
        .frame(maxWidth: .infinity, alignment: .leading)
        .sheet(isPresented: $showingFeedback) { feedbackSheet }
        #if INTERNAL_TESTING
        .sheet(isPresented: $showingInternalReview) {
            if let benchmark {
                InternalReviewForm(caption: caption, image: sourceImage, benchmark: benchmark) {
                    internalFeedbackSubmitted = true
                }
            }
        }
        #endif
    }

    private var feedbackSheet: some View {
        NavigationStack {
            Form {
                Section {
                    ForEach(FeedbackChoice.allCases, id: \.self) { choice in
                        Button { toggle(choice) } label: {
                            HStack {
                                Text(choice.title)
                                Spacer()
                                if selection.contains(choice) { Image(systemName: "checkmark") }
                            }
                        }
                        .disabled(isSaving)
                        .accessibilityAddTraits(selection.contains(choice) ? .isSelected : [])
                    }
                    if isSaving { ProgressView() }
                } header: {
                    Text("How was this caption useful?")
                } footer: {
                    Text("You can select Generally useful together with specific reasons. Not useful must be selected alone. No image or caption text is saved with your feedback.")
                }
                Button("Send feedback") { submit() }
                    .disabled(selection.isEmpty || isSaving)
            }
            .navigationTitle("Caption feedback")
            .navigationBarTitleDisplayMode(.inline)
            .interactiveDismissDisabled(isSaving)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Cancel") { showingFeedback = false }
                        .disabled(isSaving)
                }
            }
            .alert("Couldn't save feedback", isPresented: $saveError) {
                Button("OK", role: .cancel) { }
            } message: {
                Text("Please try again. Your latest change was not saved.")
            }
        }
    }

    private func toggle(_ choice: FeedbackChoice) {
        if selection.contains(choice) {
            selection.remove(choice)
        } else if choice == .notUseful {
            selection = [choice]
        } else {
            selection.remove(.notUseful)
            selection.insert(choice)
        }
    }

    private func submit() {
        guard !isSaving, submittedSelection.isEmpty, !selection.isEmpty, let benchmark else { return }
        let choices = selection
        guard choices != submittedSelection else {
            showingFeedback = false
            return
        }
        isSaving = true
        let payload = CaptionFeedbackPayload(
            benchmark_event_id: benchmark.eventID, revision: 1,
            is_useful: !choices.contains(.notUseful),
            reasons: choices.flatMap(\.reasons).sorted()
        )
        Task {
            do {
                try await BenchmarkTelemetry.shared.recordFeedback(payload, modelVersion: benchmark.modelVersion)
                submittedSelection = choices
                showingFeedback = false
            } catch {
                saveError = true
            }
            isSaving = false
        }
    }
}

private enum FeedbackChoice: String, CaseIterable {
    case notUseful
    case generallyUseful
    case overallSituation = "overall_situation"
    case identifyingObjects = "identifying_objects"
    case noticingMissedDetails = "noticing_missed_details"

    var reasons: [String] {
        switch self {
        case .notUseful, .generallyUseful: []
        default: [rawValue]
        }
    }

    var title: LocalizedStringKey {
        switch self {
        case .notUseful: "Not useful"
        case .generallyUseful: "Generally useful"
        case .overallSituation: "Understanding the overall situation"
        case .identifyingObjects: "Identifying objects"
        case .noticingMissedDetails: "Noticing things I might have missed"
        }
    }
}
