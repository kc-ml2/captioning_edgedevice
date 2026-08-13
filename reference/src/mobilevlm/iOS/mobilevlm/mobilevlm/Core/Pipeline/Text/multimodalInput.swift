// multimodalInput.swift


import Foundation
import CoreML

func buildMultimodalEmbeddings(
    inputIds: [Int],
    imageFeatures: MLMultiArray,
) -> MLMultiArray? {

    let hiddenSize = 2048
    let imageTokenIndex = -200

    let imageSeqLen: Int

    if imageFeatures.shape.count == 3 {

        imageSeqLen = imageFeatures.shape[1].intValue

    } else if imageFeatures.shape.count == 2 {

        imageSeqLen = imageFeatures.shape[0].intValue

    } else {

        print("❌ Unsupported imageFeatures shape")
        return nil
    }

    let textTokenCount =
        inputIds.filter { $0 != imageTokenIndex }.count

    let finalSeqLen =
        textTokenCount + imageSeqLen

    guard let output = try? MLMultiArray(
        shape: [
            NSNumber(value: 1),
            NSNumber(value: finalSeqLen),
            NSNumber(value: hiddenSize)
        ],
        dataType: .float32
    ) else {

        print("❌ MLMultiArray allocation failed")
        return nil
    }

    let outputPtr = output.dataPointer.bindMemory(
        to: Float32.self,
        capacity: output.count
    )

    let imagePtr = imageFeatures.dataPointer.bindMemory(
        to: Float32.self,
        capacity: imageFeatures.count
    )

    var outRow = 0

    for tokenId in inputIds {

        if tokenId == imageTokenIndex {

            for i in 0..<imageSeqLen {

                let imageOffset =
                    i * hiddenSize

                let outputOffset =
                    outRow * hiddenSize

                for j in 0..<hiddenSize {

                    outputPtr[outputOffset + j] =
                        imagePtr[imageOffset + j]
                }

                outRow += 1
            }

        } else {

            let embedding =
                makeNextTokenEmbedding(
                    tokenId: tokenId
                )

            let embPtr = embedding.dataPointer.bindMemory(
                to: Float32.self,
                capacity: embedding.count
            )

            let outputOffset =
                outRow * hiddenSize

            for j in 0..<hiddenSize {

                outputPtr[outputOffset + j] =
                    embPtr[j]
            }

            outRow += 1
        }
    }

    return output
}
