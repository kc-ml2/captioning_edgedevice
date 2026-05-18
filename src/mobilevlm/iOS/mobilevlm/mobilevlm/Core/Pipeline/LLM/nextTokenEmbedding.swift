import Foundation
import CoreML

func updateNextTokenEmbedding(
    tokenId: Int,
    embedBuffer: MLMultiArray
) {

    let hiddenDim = 2048

    let embeddingWeights =
        ModelManager.shared.embeddingWeights

    let embeddingScale: Float32 =
        0.0029681348

    let embedPtr = embedBuffer.dataPointer.bindMemory(
        to: Float32.self,
        capacity: embedBuffer.count
    )

    let startIndex = tokenId * hiddenDim

    for i in 0..<hiddenDim {

        embedPtr[i] =
            Float32(
                embeddingWeights[startIndex + i]
            ) * embeddingScale
    }
}
