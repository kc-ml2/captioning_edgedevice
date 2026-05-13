import Foundation
import CoreML




func updateNextTokenEmbedding(
    tokenId: Int,
    embedBuffer: MLMultiArray
) {

    let hiddenDim = 2048

    let embeddingWeights =
        ModelManager.shared.embeddingWeights

    let embedPtr = embedBuffer.dataPointer.bindMemory(
        to: Float32.self,
        capacity: embedBuffer.count
    )

    let startIndex = tokenId * hiddenDim

    for i in 0..<hiddenDim {

        embedPtr[i] =
            embeddingWeights[startIndex + i]
    }
}
