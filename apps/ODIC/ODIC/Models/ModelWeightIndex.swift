import Foundation

nonisolated struct ModelWeightIndex: Decodable {
    let weightMap: [String: String]

    enum CodingKeys: String, CodingKey {
        case weightMap = "weight_map"
    }

    var shards: [String] {
        Set(weightMap.values).sorted()
    }

    static func load(from directory: URL) throws -> Self {
        let url = directory.appending(path: "model.safetensors.index.json")
        return try JSONDecoder().decode(Self.self, from: Data(contentsOf: url))
    }
}
