import Foundation
import CoreML

func makePrefillAttentionMask(curPos: Int) -> MLMultiArray? {

    guard let attentionMask = try? MLMultiArray(
        shape: [
            NSNumber(value: 1),
            NSNumber(value: curPos)
        ],
        dataType: .float32
    ) else {

        print("❌ Failed to create attention mask")
        return nil
    }

    let ptr = attentionMask.dataPointer.bindMemory(
        to: Float32.self,
        capacity: attentionMask.count
    )

    for i in 0..<attentionMask.count {
        ptr[i] = 1.0
    }

    return attentionMask
}
