import Foundation
import SentencepieceTokenizer

func tokenizeImageQuestion(
    question: String
) -> [Int]? {

    
    // 매번 토크나이저 불러오지 말고 재사용하기. modelManager 참고
    
    // =========================================
    // tokenizer model
    // =========================================

    guard let modelPath = Bundle.main.path(
        forResource: "tokenizer",
        ofType: "model"
    ) else {

        print("❌ tokenizer.model not found")
        return nil
    }

    // =========================================
    // constants
    // =========================================

    let BOS_ID = 1
    let IMAGE_TOKEN_INDEX = -200

    // =========================================
    // prompt
    // =========================================

    let prompt =
        "A chat between a curious user and an artificial intelligence assistant. " +
        "The assistant gives helpful, detailed, and polite answers to the user's questions. " +
        "USER: <image>\n" +
        question +
        " ASSISTANT:"

    do {

        // =====================================
        // tokenizer
        // =====================================

        let tokenizer = try SentencepieceTokenizer(
            modelPath: modelPath
        )

        // =====================================
        // split prompt
        // =====================================

        let chunks = prompt.components(
            separatedBy: "<image>"
        )

        var inputIds: [Int] = []

        inputIds.reserveCapacity(512)

        // =====================================
        // tokenize
        // =====================================

        for chunkIndex in 0..<chunks.count {

            let encoded = try tokenizer.encode(
                chunks[chunkIndex]
            )

            // =================================
            // add BOS only once
            // =================================

            if chunkIndex == 0 {

                inputIds.append(BOS_ID)

                for token in encoded {
                    inputIds.append(Int(token) - 1)
                }

            } else {

                // image token separator
                inputIds.append(IMAGE_TOKEN_INDEX)

                for token in encoded {

                    let id = Int(token) - 1

                    // skip duplicated BOS
                    if id != BOS_ID {
                        inputIds.append(id)
                    }
                }
            }
        }

        return inputIds

    } catch {

        print("❌ tokenizer error:", error)
        return nil
    }
}
