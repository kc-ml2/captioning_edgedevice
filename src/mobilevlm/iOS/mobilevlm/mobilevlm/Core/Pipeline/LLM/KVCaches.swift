import CoreML
import Foundation


final class KVCache {

    let key: MLMultiArray
    let value: MLMultiArray

    init(
        key: MLMultiArray,
        value: MLMultiArray
    ) {
        self.key = key
        self.value = value
    }
}

func createKVCache(seqLen: Int) -> MLMultiArray {

    let kvCache = try! MLMultiArray(
        shape: [
            NSNumber(value: 48),
            NSNumber(value: 16),
            NSNumber(value: seqLen),
            NSNumber(value: 128)
        ],
        dataType: .float32
    )

    memset(
        kvCache.dataPointer,
        0,
        kvCache.count * MemoryLayout<Float32>.size
    )

    return kvCache
}


func copyKVToBuffer(
    src: MLMultiArray,
    dst: MLMultiArray,
    length: Int
) {

    let numKV = 48
    let numHeads = 16
    let headDim = 128
    let dstMaxLen = dst.shape[2].intValue

    let srcPtr = src.dataPointer.bindMemory(
        to: Float32.self,
        capacity: src.count
    )

    let dstPtr = dst.dataPointer.bindMemory(
        to: Float32.self,
        capacity: dst.count
    )

    for kv in 0..<numKV {
        for head in 0..<numHeads {

            let srcOffset =
                ((kv * numHeads + head)
                 * length
                 * headDim)

            let dstOffset =
                ((kv * numHeads + head)
                 * dstMaxLen
                 * headDim)

            memcpy(
                dstPtr.advanced(by: dstOffset),
                srcPtr.advanced(by: srcOffset),
                length * headDim * MemoryLayout<Float32>.size
            )
        }
    }
}

func appendLastKV(
    src: MLMultiArray,
    dst: MLMultiArray,
    position: Int
) {

    let numKV = 48
    let numHeads = 16
    let headDim = 128

    let srcLen = src.shape[2].intValue
    let dstMaxLen = dst.shape[2].intValue

    let srcPtr = src.dataPointer.bindMemory(
        to: Float32.self,
        capacity: src.count
    )

    let dstPtr = dst.dataPointer.bindMemory(
        to: Float32.self,
        capacity: dst.count
    )

    for kv in 0..<numKV {
        for head in 0..<numHeads {

            let srcBase =
                ((kv * numHeads + head)
                 * srcLen
                 * headDim)

            let dstBase =
                ((kv * numHeads + head)
                 * dstMaxLen
                 * headDim)

            memcpy(
                dstPtr.advanced(
                    by: dstBase + position * headDim
                ),
                srcPtr.advanced(
                    by: srcBase + (srcLen - 1) * headDim
                ),
                headDim * MemoryLayout<Float32>.size
            )
        }
    }
}
