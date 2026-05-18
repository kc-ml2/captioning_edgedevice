import CoreML

func getNextToken(
    logits: MLMultiArray
) -> Int {

    let seqLen = logits.shape[1].intValue
    let vocabSize = logits.shape[2].intValue

    let lastSeqIndex = seqLen - 1

    let ptr = logits.dataPointer.bindMemory(
        to: Float32.self,
        capacity: logits.count
    )

    let strides = logits.strides.map {
        $0.intValue
    }

    var maxValue: Float32 = -.infinity
    var nextToken = 0

    // =========================
    // Argmax
    // =========================

    for vocabIndex in 0..<vocabSize {

        let index =
            0 * strides[0] +
            lastSeqIndex * strides[1] +
            vocabIndex * strides[2]

        let value = ptr[index]

        if value > maxValue {

            maxValue = value
            nextToken = vocabIndex
        }
    }

    return nextToken
}
