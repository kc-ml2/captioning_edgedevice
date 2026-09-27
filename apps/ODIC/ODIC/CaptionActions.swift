import AVFoundation
import Combine
import SwiftUI

/// One instance per displayed caption. Never persists the caption or image.
struct CaptionActions: View {
    let caption: String
    let benchmark: BenchmarkReference?
    @StateObject private var speech = CaptionSpeech()
    @State private var selection: Set<FeedbackChoice> = []
    @State private var submittedSelection: Set<FeedbackChoice> = []
    @State private var revision = 0
    @State private var isSaving = false
    @State private var showingFeedback = false
    @State private var saveError = false
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        HStack {
            Button { speech.toggle(caption) } label: {
                Label(speech.isSpeaking ? "Stop reading" : "Read aloud",
                      systemImage: speech.isSpeaking ? "stop.fill" : "speaker.wave.2.fill")
            }
            Button {
                selection = submittedSelection
                showingFeedback = true
            } label: {
                Label("Feedback", systemImage: submittedSelection.isEmpty ? "bubble.left" : "checkmark.bubble")
            }
            .disabled(benchmark == nil)
        }
        .buttonStyle(.bordered)
        .tint(.white)
        .frame(maxWidth: .infinity, alignment: .leading)
        .sheet(isPresented: $showingFeedback) { feedbackSheet }
        .onChange(of: scenePhase) { _, phase in
            if phase != .active { speech.stop() }
        }
        .onDisappear { speech.stop() }
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
                    Text("Select multiple specific reasons, or choose Not useful or Generally useful alone. No image or caption text is saved with your feedback.")
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
        } else if choice == .notUseful || choice == .generallyUseful {
            selection = [choice]
        } else {
            selection.remove(.notUseful)
            selection.remove(.generallyUseful)
            selection.insert(choice)
        }
    }

    private func submit() {
        guard !isSaving, !selection.isEmpty, let benchmark else { return }
        let choices = selection
        guard choices != submittedSelection else {
            showingFeedback = false
            return
        }
        isSaving = true
        let nextRevision = revision + 1
        let payload = CaptionFeedbackPayload(
            benchmark_event_id: benchmark.eventID, revision: nextRevision,
            is_useful: !choices.contains(.notUseful),
            reasons: choices.flatMap(\.reasons).sorted()
        )
        Task {
            do {
                try await BenchmarkTelemetry.shared.recordFeedback(payload, modelVersion: benchmark.modelVersion)
                revision = nextRevision
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

@MainActor
private final class CaptionSpeech: NSObject, ObservableObject, AVSpeechSynthesizerDelegate {
    @Published private(set) var isSpeaking = false
    private let synthesizer = AVSpeechSynthesizer()

    override init() {
        super.init()
        synthesizer.delegate = self
    }

    func toggle(_ text: String) {
        if isSpeaking {
            stop()
            return
        }
        // Captions currently use English regardless of the UI language.
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        isSpeaking = true
        synthesizer.speak(utterance)
    }

    func stop() {
        synthesizer.stopSpeaking(at: .immediate)
        isSpeaking = false
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        Task { @MainActor [weak self] in
            guard let self else { return }
            self.isSpeaking = self.synthesizer.isSpeaking
        }
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        Task { @MainActor [weak self] in
            guard let self else { return }
            self.isSpeaking = self.synthesizer.isSpeaking
        }
    }
}
