import AVFoundation
import Combine
import SwiftUI

/// One instance per displayed caption. Never persists the caption or image.
struct CaptionActions: View {
    let caption: String
    let benchmark: BenchmarkReference?
    var isGenerating = false
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
    @Environment(\.scenePhase) private var scenePhase

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
        .onAppear { readAutomaticallyIfNeeded() }
        .onChange(of: caption) { _, _ in readAutomaticallyIfNeeded() }
        .onChange(of: isGenerating) { _, _ in readAutomaticallyIfNeeded() }
        .onChange(of: scenePhase) { _, phase in
            if phase == .active { readAutomaticallyIfNeeded() }
            else { speech.stop() }
        }
        .onDisappear { speech.stop() }
    }

    private func readAutomaticallyIfNeeded() {
        guard scenePhase == .active, !showingFeedback else { return }
        speech.appendStreaming(caption, final: !isGenerating)
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

@MainActor
final class CaptionSpeech: NSObject, ObservableObject, AVSpeechSynthesizerDelegate {
    @Published private(set) var isSpeaking = false
    @Published private(set) var outputVolume: Float = AVAudioSession.sharedInstance().outputVolume
    private var volumeObservation: NSKeyValueObservation?
    private let synthesizer = AVSpeechSynthesizer()
    private var spokenPrefix = ""
    private var streamingStopped = false

    func beginCaption() {
        stop()
        spokenPrefix = ""
        streamingStopped = false
    }

    func appendStreaming(_ text: String, final: Bool) {
        guard !streamingStopped else { return }
        // Never repeat already queued speech if the tokenizer revises a suffix.
        guard text.hasPrefix(spokenPrefix) else {
            streamingStopped = true
            return
        }
        let remainder = String(text.dropFirst(spokenPrefix.count))
        var boundary: String.Index?
        if final {
            boundary = remainder.endIndex
        } else if let punctuation = remainder.lastIndex(where: { ".!?;:".contains($0) }) {
            boundary = remainder.index(after: punctuation)
        } else if remainder.split(whereSeparator: \.isWhitespace).count >= 9 {
            // Keep the last (potentially incomplete) word until the next update.
            boundary = remainder.lastIndex(where: \.isWhitespace)
        }
        guard let boundary else { return }
        let chunk = String(remainder[..<boundary])
        guard !chunk.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        spokenPrefix += chunk
        speak(chunk)
    }

    override init() {
        super.init()
        synthesizer.delegate = self
        // Ambient respects the silent switch. iOS does not expose its state;
        // outputVolume alone must not be presented as proof of audible output.
        synthesizer.usesApplicationAudioSession = true
        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.ambient, mode: .default)
            try session.setActive(true)
        } catch {
            // Retain system speech behavior if session configuration is unavailable.
        }
        volumeObservation = session.observe(\.outputVolume, options: [.initial, .new]) { [weak self] session, _ in
            let volume = session.outputVolume
            Task { @MainActor [weak self] in self?.outputVolume = volume }
        }
    }

    func toggle(_ text: String) {
        if isSpeaking {
            stop()
            return
        }
        streamingStopped = true
        speak(text)
    }

    func speak(_ text: String) {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        // Captions currently use English regardless of the UI language.
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        isSpeaking = true
        synthesizer.speak(utterance)
    }

    func stop() {
        streamingStopped = true
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
