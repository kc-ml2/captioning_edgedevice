import CoreML

final class KVCache {

    let key: MLMultiArray
    let value: MLMultiArray

    var validLength: Int = 0

    init(
        key: MLMultiArray,
        value: MLMultiArray
    ) {
        self.key = key
        self.value = value
    }
}

final class AppBufferManager {

    static let shared = AppBufferManager()

    let kvCaches: [KVCache]

    let reusableNextEmbed: MLMultiArray

    private init() {

        // =========================
        // KV cache
        // =========================

        let kvShape: [NSNumber] = [
            1, 16, 1, 128
        ]

        func makeArray(
            shape: [NSNumber]
        ) -> MLMultiArray {

            guard let array = try? MLMultiArray(
                shape: shape,
                dataType: .float32
            ) else {
                fatalError("Failed to create MLMultiArray")
            }

            let ptr = array.dataPointer.bindMemory(
                to: Float32.self,
                capacity: array.count
            )

            ptr.initialize(
                repeating: 0,
                count: array.count
            )

            return array
        }

        self.kvCaches = (0..<24).map { _ in

            KVCache(
                key: makeArray(shape: kvShape),
                value: makeArray(shape: kvShape)
            )
        }

        // =========================
        // reusable next embed
        // =========================

        guard let embed = try? MLMultiArray(
            shape: [1, 1, 2048],
            dataType: .float32
        ) else {
            fatalError("Failed to create embed buffer")
        }

        let embedPtr = embed.dataPointer.bindMemory(
            to: Float32.self,
            capacity: embed.count
        )

        embedPtr.initialize(
            repeating: 0,
            count: embed.count
        )

        self.reusableNextEmbed = embed
    }

    func resetKVCache() {

        for cache in kvCaches {

            cache.validLength = 0

            zero(cache.key)
            zero(cache.value)
        }
    }

    private func zero(
        _ array: MLMultiArray
    ) {

        let ptr = array.dataPointer.bindMemory(
            to: Float32.self,
            capacity: array.count
        )

        ptr.update(
            repeating: 0,
            count: array.count
        )
    }
}
