import CoreML

func getNextToken(logits: MLMultiArray) -> Int {

    let vocabSize = 32000

    let ptr = logits.dataPointer.bindMemory(
        to: Float32.self,
        capacity: vocabSize
    )

    var maxValue = ptr[0]
    var maxIndex = 0

    for i in 1..<vocabSize {

        let value = ptr[i]

        if value > maxValue {
            maxValue = value
            maxIndex = i
        }
    }

    return maxIndex
}
