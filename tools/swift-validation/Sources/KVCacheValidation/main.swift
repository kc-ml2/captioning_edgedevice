import Foundation
import MLX
import MLXNN

let hiddenSize = 2048
let heads = 16
let headDimension = 128
let layers = 24
let imageToken: Int32 = -200

struct KVCache { var key: MLXArray; var value: MLXArray }
struct Failure: Error, CustomStringConvertible { let description: String }

func require(_ arrays: [String: MLXArray], _ name: String) throws -> MLXArray {
    guard let value = arrays[name] else { throw Failure(description: "Missing: \(name)") }
    return value
}

func linear(_ x: MLXArray, _ weights: [String: MLXArray], _ name: String) throws -> MLXArray {
    x.matmul(try require(weights, name).transposed())
}

func norm(_ x: MLXArray, _ weight: MLXArray) -> MLXArray {
    x * rsqrt((x * x).mean(axis: -1, keepDims: true) + 1e-6) * weight
}

func rotateHalf(_ x: MLXArray) -> MLXArray {
    concatenated([
        -x[0..., 0..., 0..., (headDimension / 2)...],
        x[0..., 0..., 0..., ..<(headDimension / 2)],
    ], axis: -1)
}

func rope(_ x: MLXArray, positions: MLXArray) -> MLXArray {
    let dimensions = arange(0, headDimension, step: 2).asType(.float32)
    let inverse = 1.0 / pow(10_000.0, dimensions / Float(headDimension))
    let frequencies = outer(positions.asType(.float32), inverse)
    let embedding = concatenated([frequencies, frequencies], axis: -1)
        .reshaped(1, 1, positions.size, headDimension)
    return x * cos(embedding) + rotateHalf(x) * sin(embedding)
}

func decoderLayer(
    _ input: MLXArray,
    _ weights: [String: MLXArray],
    layer: Int,
    positions: MLXArray,
    past: KVCache?
) throws -> (MLXArray, KVCache) {
    let prefix = "language_model.layers.\(layer)"
    let length = input.shape[1]
    let normalized = norm(input, try require(weights, "\(prefix).input_layernorm.weight"))
    let q = rope(
        try linear(normalized, weights, "\(prefix).self_attn.q_proj.weight")
            .reshaped(1, length, heads, headDimension).transposed(0, 2, 1, 3),
        positions: positions
    )
    let newK = rope(
        try linear(normalized, weights, "\(prefix).self_attn.k_proj.weight")
            .reshaped(1, length, heads, headDimension).transposed(0, 2, 1, 3),
        positions: positions
    )
    let newV = try linear(normalized, weights, "\(prefix).self_attn.v_proj.weight")
        .reshaped(1, length, heads, headDimension).transposed(0, 2, 1, 3)
    let key = past.map { concatenated([$0.key, newK], axis: 2) } ?? newK
    let value = past.map { concatenated([$0.value, newV], axis: 2) } ?? newV
    var scores = q.matmul(key.transposed(0, 1, 3, 2)) / sqrt(Float(headDimension))
    if past == nil {
        scores = scores + (1.0 - tri(length, dtype: .float32)) * -1e9
    }
    var attention = softmax(scores, axis: -1).matmul(value).transposed(0, 2, 1, 3)
        .reshaped(1, length, hiddenSize)
    attention = try linear(attention, weights, "\(prefix).self_attn.o_proj.weight")
    let residual = input + attention
    let normalized2 = norm(
        residual, try require(weights, "\(prefix).post_attention_layernorm.weight")
    )
    let gate = silu(try linear(normalized2, weights, "\(prefix).mlp.gate_proj.weight"))
    let up = try linear(normalized2, weights, "\(prefix).mlp.up_proj.weight")
    let output = residual + (try linear(gate * up, weights, "\(prefix).mlp.down_proj.weight"))
    return (output, KVCache(key: key, value: value))
}

func compare(_ name: String, _ actual: MLXArray, _ expected: MLXArray, tolerance: Double) -> Bool {
    let difference = abs(actual.asType(.float32) - expected.asType(.float32))
    let maxError = difference.max(); let meanError = difference.mean()
    let matches = allClose(actual, expected, rtol: tolerance, atol: tolerance)
    eval(maxError, meanError, matches)
    print("\(name): passed=\(matches.item(Bool.self)) max_abs=\(maxError.item(Float.self)) mean_abs=\(meanError.item(Float.self)) shape=\(actual.shape)")
    return matches.item(Bool.self)
}

func run(_ url: URL) throws {
    let weights = try loadArrays(url: url)
    let ids = try require(weights, "input.ids").asArray(Int32.self)
    guard let imagePosition = ids.firstIndex(of: imageToken) else { throw Failure(description: "No image token") }
    let embedding = try require(weights, "language_model.embed_tokens.weight")
    let before = embedding.take(MLXArray(ids[..<imagePosition].map(Int.init)), axis: 0)
    let after = embedding.take(MLXArray(ids[(imagePosition + 1)...].map(Int.init)), axis: 0)
    let image = try require(weights, "input.image_features")
    let multimodal = concatenated([before, image, after], axis: 0).expandedDimensions(axis: 0)
    var passed = compare("multimodal", multimodal, try require(weights, "expected.multimodal_embeddings"), tolerance: 0)

    let prefillLength = multimodal.shape[1]
    var hidden = multimodal
    var caches: [KVCache] = []
    for layer in 0 ..< layers {
        let result = try decoderLayer(
            hidden, weights, layer: layer, positions: arange(prefillLength), past: nil
        )
        hidden = result.0; caches.append(result.1); eval(hidden)
        passed = compare("prefill.cache.\(layer).key", result.1.key, try require(weights, "expected.prefill_cache.\(layer).key"), tolerance: 2e-2) && passed
        passed = compare("prefill.cache.\(layer).value", result.1.value, try require(weights, "expected.prefill_cache.\(layer).value"), tolerance: 2e-2) && passed
    }
    passed = compare("prefill.hidden", hidden, try require(weights, "expected.prefill_hidden"), tolerance: 3e-2) && passed
    let prefillLogits = try linear(norm(hidden, try require(weights, "language_model.norm.weight")), weights, "lm_head.weight")[0, prefillLength - 1]
    passed = compare("prefill.logits", prefillLogits, try require(weights, "expected.prefill_logits")[0], tolerance: 5e-2) && passed
    let firstToken = prefillLogits.argMax().item(Int32.self)
    let expectedFirst = try require(weights, "expected.first_token").item(Int32.self)
    print("first_token: actual=\(firstToken) expected=\(expectedFirst)")

    var decodeHidden = embedding[Int(firstToken)].reshaped(1, 1, hiddenSize)
    var decodeCaches: [KVCache] = []
    for layer in 0 ..< layers {
        let result = try decoderLayer(
            decodeHidden,
            weights,
            layer: layer,
            positions: MLXArray([prefillLength]),
            past: caches[layer]
        )
        decodeHidden = result.0; decodeCaches.append(result.1); eval(decodeHidden)
    }
    passed = compare("decode.hidden", decodeHidden, try require(weights, "expected.decode_hidden"), tolerance: 3e-2) && passed
    for layer in [0, layers - 1] {
        passed = compare("decode.cache.\(layer).key", decodeCaches[layer].key, try require(weights, "expected.decode_cache.\(layer).key"), tolerance: 2e-2) && passed
        passed = compare("decode.cache.\(layer).value", decodeCaches[layer].value, try require(weights, "expected.decode_cache.\(layer).value"), tolerance: 2e-2) && passed
    }
    let decodeLogits = try linear(norm(decodeHidden, try require(weights, "language_model.norm.weight")), weights, "lm_head.weight")[0, 0]
    passed = compare("decode.logits", decodeLogits, try require(weights, "expected.decode_logits")[0], tolerance: 5e-2) && passed
    let secondToken = decodeLogits.argMax().item(Int32.self)
    let expectedSecond = try require(weights, "expected.second_token").item(Int32.self)
    print("second_token: actual=\(secondToken) expected=\(expectedSecond)")

    guard passed && firstToken == expectedFirst && secondToken == expectedSecond else {
        throw Failure(description: "Multimodal prefill/KV decode differs from PyTorch")
    }
    print("PASS: MLX Swift multimodal prefill and one-token KV-cache decode match PyTorch.")
}

do {
    guard CommandLine.arguments.count == 2 else { throw Failure(description: "Usage: KVCacheValidation <fixture>") }
    try run(URL(filePath: CommandLine.arguments[1]))
} catch {
    FileHandle.standardError.write(Data("ERROR: \(error)\n".utf8)); exit(1)
}
