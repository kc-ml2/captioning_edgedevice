import AVFoundation
import Combine
import OSLog

@MainActor
final class CaptionSpeech: NSObject, ObservableObject, AVSpeechSynthesizerDelegate {
    @Published private(set) var isSpeaking = false
    @Published private(set) var outputVolume: Float = AVAudioSession.sharedInstance().outputVolume
    private var volumeObservation: NSKeyValueObservation?
    private var interruptionObservation: NSObjectProtocol?
    private let synthesizer = AVSpeechSynthesizer()
    private let logger = Logger(subsystem: "com.ml2.scenesense", category: "CaptionSpeech")
    private var completedPrefix = ""
    private var latestText = ""
    private var final = false
    private var userStopped = false
    private var suspended = false
    private var interrupted = false
    private var current: AVSpeechUtterance?
    private var currentChunk = ""

    override init() {
        super.init()
        synthesizer.delegate = self
        synthesizer.usesApplicationAudioSession = true
        volumeObservation = AVAudioSession.sharedInstance().observe(\.outputVolume, options: [.initial, .new]) { [weak self] session, _ in
            let volume = session.outputVolume
            Task { @MainActor [weak self] in self?.outputVolume = volume }
        }
        interruptionObservation = NotificationCenter.default.addObserver(
            forName: AVAudioSession.interruptionNotification, object: nil, queue: .main
        ) { [weak self] notification in
            let began = (notification.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt) == AVAudioSession.InterruptionType.began.rawValue
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.interrupted = began
                if began { self.cancelCurrent() } else { self.pump() }
            }
        }
    }

    deinit {
        if let interruptionObservation { NotificationCenter.default.removeObserver(interruptionObservation) }
    }

    func beginCaption() {
        cancelCurrent()
        completedPrefix = ""
        latestText = ""
        final = false
        userStopped = false
    }

    func appendStreaming(_ text: String, final: Bool) {
        latestText = text
        self.final = final
        pump()
    }

    func suspend() { suspended = true; cancelCurrent() }
    func resume() { suspended = false; pump() }
    func stop() { userStopped = true; cancelCurrent() }

    func toggle(_ text: String) {
        if isSpeaking { stop(); return }
        beginCaption()
        appendStreaming(text, final: true)
    }

    private func cancelCurrent() {
        // Ignore callbacks from canceled utterances; completedPrefix only tracks finished phrases.
        current = nil
        currentChunk = ""
        synthesizer.stopSpeaking(at: .immediate)
        isSpeaking = false
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }

    private func pump() {
        guard !userStopped, !suspended, !interrupted, current == nil,
              let chunk = SpeechChunks.next(text: latestText, completedPrefix: completedPrefix, final: final) else { return }
        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.playback, mode: .default, options: [.mixWithOthers])
            try session.setActive(true)
            outputVolume = session.outputVolume
        } catch {
            // Do not consume the phrase: the next update/resume can retry it.
            logger.error("Speech audio session activation failed: \(error.localizedDescription, privacy: .public)")
            return
        }
        let utterance = AVSpeechUtterance(string: chunk)
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        current = utterance
        currentChunk = chunk
        isSpeaking = true
        synthesizer.speak(utterance)
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        let id = ObjectIdentifier(utterance)
        Task { @MainActor [weak self] in
            guard let self, let current = self.current, ObjectIdentifier(current) == id else { return }
            self.completedPrefix += self.currentChunk
            self.current = nil
            self.currentChunk = ""
            self.isSpeaking = false
            self.pump()
            if self.current == nil {
                try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
            }
        }
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        let id = ObjectIdentifier(utterance)
        Task { @MainActor [weak self] in
            guard let self, let current = self.current, ObjectIdentifier(current) == id else { return }
            self.current = nil
            self.currentChunk = ""
            self.isSpeaking = false
        }
    }
}
