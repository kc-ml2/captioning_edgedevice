import Foundation
import SentencepieceTokenizer

func decodeTokens(
    generatedTokens: [Int]
) -> String? {

    do {

        let tokenizer =
            ModelManager.shared.tokenizer

        let text = try tokenizer.decode(
            generatedTokens.map { $0 + 1 }
        )

        let caption = text
            .replacingOccurrences(
                of: "\\s+",
                with: " ",
                options: NSString.CompareOptions.regularExpression
            )
            .trimmingCharacters(
                in: CharacterSet.whitespacesAndNewlines
            )

        return caption

    } catch {

        print("❌ decode failed:", error)
        return nil
    }
}
