import Foundation
import SentencepieceTokenizer

func decodeTokens(
    generatedTokens: [Int]
) -> String? {

    guard let path = Bundle.main.path(
        forResource: "tokenizer",
        ofType: "model"
    ) else {

        print("❌ tokenizer.model not found")
        return nil
    }

    do {

        let tokenizer = try SentencepieceTokenizer(
            modelPath: path
        )

        // restore original token ids
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
