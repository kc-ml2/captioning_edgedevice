//import CoreML
//
//struct KVCache {
//    let key: MLMultiArray
//    let value: MLMultiArray
//    var validLength: Int
//}
//
//func createEmptyKVCache() -> [KVCache] {
//
//    let shape: [NSNumber] = [1, 16, 1, 128]
//
//    func makeArray() -> MLMultiArray {
//
//        guard let array = try? MLMultiArray(
//            shape: shape,
//            dataType: .float32
//        ) else {
//            fatalError("Failed to create MLMultiArray")
//        }
//
//        let ptr = array.dataPointer.bindMemory(
//            to: Float32.self,
//            capacity: array.count
//        )
//
//        ptr.initialize(repeating: 0, count: array.count)
//
//        return array
//    }
//
//    return (0..<24).map { _ in
//
//        KVCache(
//            key: makeArray(),
//            value: makeArray(),
//            validLength: 0
//        )
//    }
//}
