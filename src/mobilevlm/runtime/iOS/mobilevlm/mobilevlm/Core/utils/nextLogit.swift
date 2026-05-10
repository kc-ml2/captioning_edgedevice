import CoreML

func getNextToken(
    logits: MLMultiArray
) -> Int {

    // logits shape:
    // [1, seq_len, vocab_size]

    let seqLen = logits.shape[1].intValue
    let vocabSize = logits.shape[2].intValue

    let lastSeqIndex = seqLen - 1

    let ptr = logits.dataPointer.bindMemory(
        to: Float32.self,
        capacity: logits.count
    )

    var maxValue: Float32 = -Float.infinity
    var nextToken: Int = 0

    for vocabIndex in 0..<vocabSize {

        // [batch=0, seq=last, vocab=vocabIndex]

        let flatIndex =
            lastSeqIndex * vocabSize + vocabIndex

        let value = ptr[flatIndex]

        if value > maxValue {

            maxValue = value
            nextToken = vocabIndex
        }
    }

    return nextToken
}
