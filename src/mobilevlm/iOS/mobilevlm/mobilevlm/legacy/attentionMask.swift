//import Foundation
//import CoreML
//
//func makePrefillAttentionMask(
//    maxSeqLen: Int,
//    curLen: Int
//) -> MLMultiArray? {
//
//    guard let attentionMask = try? MLMultiArray(
//        shape: [
//            NSNumber(value: 1),
//            NSNumber(value: maxSeqLen)
//        ],
//        dataType: .float32
//    ) else {
//
//        print("❌ Failed to create attention mask")
//        return nil
//    }
//
//    let ptr = attentionMask.dataPointer.bindMemory(
//        to: Float32.self,
//        capacity: attentionMask.count
//    )
//
//    for i in 0..<maxSeqLen {
//        ptr[i] = (i < curLen) ? 1.0 : 0.0
//    }
//
//    return attentionMask
//}
