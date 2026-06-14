import Foundation
import SentencepieceTokenizer

func tokenizeImageQuestion(
    question: String
) -> [Int]? {

    let tokenizer =
        ModelManager.shared.tokenizer

    let BOS_ID = 1
    let IMAGE_TOKEN_INDEX = -200

    let prompt =
        "A chat between a curious user and an artificial intelligence assistant. " +
        "The assistant gives helpful, detailed, and polite answers to the user's questions. " +
        "USER: <image>\n" +
        question +
        " ASSISTANT:"

    let chunks = prompt.components(
        separatedBy: "<image>"
    )

    var inputIds: [Int] = []
    inputIds.reserveCapacity(512)

    do {

        for chunkIndex in 0..<chunks.count {

            let encoded =
                try tokenizer.encode(
                    chunks[chunkIndex]
                )

            if chunkIndex == 0 {

                inputIds.append(BOS_ID)

                for token in encoded {
                    inputIds.append(
                        Int(token) - 1
                    )
                }

            } else {

                inputIds.append(
                    IMAGE_TOKEN_INDEX
                )

                for token in encoded {

                    let id =
                        Int(token) - 1

                    if id != BOS_ID {
                        inputIds.append(id)
                    }
                }
            }
        }

        return inputIds

    } catch {

        print(
            "❌ Tokenization failed:",
            error
        )

        return nil
    }
}
