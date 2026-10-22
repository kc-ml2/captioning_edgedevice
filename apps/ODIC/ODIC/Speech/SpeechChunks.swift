import Foundation

nonisolated enum SpeechChunks {
    static func next(text: String, completedPrefix: String, final: Bool) -> String? {
        guard text.hasPrefix(completedPrefix) else { return nil }
        let remainder = String(text.dropFirst(completedPrefix.count))
        let boundary: String.Index?
        if let punctuation = remainder.firstIndex(where: { ".!?;:".contains($0) }) {
            boundary = remainder.index(after: punctuation)
        } else if final {
            boundary = remainder.endIndex
        } else if remainder.split(whereSeparator: \.isWhitespace).count >= 9 {
            boundary = remainder.lastIndex(where: \.isWhitespace)
        } else { boundary = nil }
        guard let boundary else { return nil }
        let chunk = String(remainder[..<boundary])
        return chunk.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : chunk
    }
}
