import Foundation
import CoreML

func makeNextTokenEmbedding(
    tokenId: Int
) -> MLMultiArray {

    let tokenInput = try! MLMultiArray(
        shape: [1, 1],
        dataType: .int32
    )

    tokenInput[0] = NSNumber(value: Int32(tokenId))

    let result =
        try! ModelManager.shared.embeddingModel.prediction(
            token: tokenInput
        )

    return result.embedding_input
}
