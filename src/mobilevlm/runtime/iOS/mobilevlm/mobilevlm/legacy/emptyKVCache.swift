import CoreML

struct KVCache {
    var key: MLMultiArray
    var value: MLMultiArray
    var validLength: Int
}

func createEmptyKVCache(
    batchSize: Int = 1,
    numLayers: Int = 24,
    numKVHeads: Int = 16,
    headDim: Int = 128
) -> [KVCache] {

    var caches: [KVCache] = []

    for _ in 0..<numLayers {

        let shape: [NSNumber] = [
            NSNumber(value: batchSize),
            NSNumber(value: numKVHeads),
            NSNumber(value: 1),
            NSNumber(value: headDim)
        ]

        guard let key = try? MLMultiArray(
            shape: shape,
            dataType: .float32
        ) else {

            fatalError("Failed to create key cache")
        }

        guard let value = try? MLMultiArray(
            shape: shape,
            dataType: .float32
        ) else {

            fatalError("Failed to create value cache")
        }

        let keyPtr = key.dataPointer.bindMemory(
            to: Float32.self,
            capacity: key.count
        )

        keyPtr.initialize(repeating: 0, count: key.count)

        let valuePtr = value.dataPointer.bindMemory(
            to: Float32.self,
            capacity: value.count
        )

        valuePtr.initialize(repeating: 0, count: value.count)

        caches.append(
            KVCache(
                key: key,
                value: value,
                validLength: 0
            )
        )
    }

    return caches
}
