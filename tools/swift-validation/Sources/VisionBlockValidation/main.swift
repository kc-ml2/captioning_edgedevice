import Foundation
import MLX
import MLXNN

struct ValidationFailure: Error, CustomStringConvertible {
    let description: String
}

func require(_ arrays: [String: MLXArray], _ name: String) throws -> MLXArray {
    guard let value = arrays[name] else {
        throw ValidationFailure(description: "Missing fixture tensor: \(name)")
    }
    return value
}

func linear(_ input: MLXArray, weights: [String: MLXArray], prefix: String) throws -> MLXArray {
    input.matmul(try require(weights, "\(prefix).weight").transposed())
        + (try require(weights, "\(prefix).bias"))
}

func layerNorm(_ input: MLXArray, weights: [String: MLXArray], prefix: String) throws -> MLXArray {
    let mean = input.mean(axis: -1, keepDims: true)
    let centered = input - mean
    let variance = (centered * centered).mean(axis: -1, keepDims: true)
    let normalized = centered * rsqrt(variance + 1e-5)
    return normalized * (try require(weights, "\(prefix).weight"))
        + (try require(weights, "\(prefix).bias"))
}

func quickGELU(_ input: MLXArray) -> MLXArray {
    input * sigmoid(1.702 * input)
}

func report(_ name: String, actual: MLXArray, expected: MLXArray, atol: Double, rtol: Double) -> Bool {
    let difference = abs(actual.asType(.float32) - expected.asType(.float32))
    let maxAbsoluteError = difference.max()
    let meanAbsoluteError = difference.mean()
    let matches = allClose(actual, expected, rtol: rtol, atol: atol)
    eval(maxAbsoluteError, meanAbsoluteError, matches)
    let passed = matches.item(Bool.self)
    print(
        "\(name): passed=\(passed) max_abs=\(maxAbsoluteError.item(Float.self)) "
            + "mean_abs=\(meanAbsoluteError.item(Float.self)) shape=\(actual.shape)"
    )
    return passed
}

func run(fixtureURL: URL) throws {
    let weights = try loadArrays(url: fixtureURL)
    let root = "vision_model.vision_model"
    let layer = "\(root).encoder.layers.0"

    let pixels = try require(weights, "input.pixels_nhwc")
    let patches = conv2d(
        pixels,
        try require(weights, "\(root).embeddings.patch_embedding.weight"),
        stride: 14
    ).reshaped(1, 576, 1024)
    let cls = try require(weights, "\(root).embeddings.class_embedding")
        .reshaped(1, 1, 1024)
    let embeddings = concatenated([cls, patches], axis: 1)
        + (try require(weights, "\(root).embeddings.position_embedding.weight"))
    let preNorm = try layerNorm(embeddings, weights: weights, prefix: "\(root).pre_layrnorm")

    let norm1 = try layerNorm(preNorm, weights: weights, prefix: "\(layer).layer_norm1")
    var q = try linear(norm1, weights: weights, prefix: "\(layer).self_attn.q_proj")
    var k = try linear(norm1, weights: weights, prefix: "\(layer).self_attn.k_proj")
    var v = try linear(norm1, weights: weights, prefix: "\(layer).self_attn.v_proj")
    q = q.reshaped(1, 577, 16, 64).transposed(0, 2, 1, 3)
    k = k.reshaped(1, 577, 16, 64).transposed(0, 2, 1, 3)
    v = v.reshaped(1, 577, 16, 64).transposed(0, 2, 1, 3)

    let scores = q.matmul(k.transposed(0, 1, 3, 2)) / sqrt(Float(64))
    let probabilities = softmax(scores, axis: -1)
    var attention = probabilities.matmul(v).transposed(0, 2, 1, 3).reshaped(1, 577, 1024)
    attention = try linear(attention, weights: weights, prefix: "\(layer).self_attn.out_proj")
    let attentionResidual = preNorm + attention

    let norm2 = try layerNorm(attentionResidual, weights: weights, prefix: "\(layer).layer_norm2")
    var hidden = try linear(norm2, weights: weights, prefix: "\(layer).mlp.fc1")
    hidden = quickGELU(hidden)
    let mlp = try linear(hidden, weights: weights, prefix: "\(layer).mlp.fc2")
    let output = attentionResidual + mlp

    let checks = [
        report("patch_tokens", actual: patches, expected: try require(weights, "expected.patch_tokens"), atol: 2e-4, rtol: 2e-4),
        report("embeddings", actual: embeddings, expected: try require(weights, "expected.embeddings"), atol: 3e-4, rtol: 3e-4),
        report("pre_norm", actual: preNorm, expected: try require(weights, "expected.pre_norm"), atol: 5e-4, rtol: 5e-4),
        report("norm1", actual: norm1, expected: try require(weights, "expected.norm1"), atol: 7e-4, rtol: 7e-4),
        report("q", actual: q, expected: try require(weights, "expected.q"), atol: 1e-3, rtol: 1e-3),
        report("k", actual: k, expected: try require(weights, "expected.k"), atol: 1e-3, rtol: 1e-3),
        report("v", actual: v, expected: try require(weights, "expected.v"), atol: 1e-3, rtol: 1e-3),
        report("attention", actual: attention, expected: try require(weights, "expected.attention"), atol: 2e-3, rtol: 2e-3),
        report("attention.residual", actual: attentionResidual, expected: try require(weights, "expected.attention_residual"), atol: 2e-3, rtol: 2e-3),
        report("norm2", actual: norm2, expected: try require(weights, "expected.norm2"), atol: 2e-3, rtol: 2e-3),
        report("mlp.hidden", actual: hidden, expected: try require(weights, "expected.mlp_hidden"), atol: 3e-3, rtol: 3e-3),
        report("block.output", actual: output, expected: try require(weights, "expected.output"), atol: 4e-3, rtol: 4e-3),
    ]

    guard checks.allSatisfy({ $0 }) else {
        throw ValidationFailure(description: "MLX Swift CLIP block differs from PyTorch reference")
    }
    print("PASS: MLX Swift CLIP embeddings and transformer block 0 match PyTorch.")
}

do {
    guard CommandLine.arguments.count == 2 else {
        throw ValidationFailure(
            description: "Usage: VisionBlockValidation <vision-fixture.safetensors>"
        )
    }
    try run(fixtureURL: URL(filePath: CommandLine.arguments[1]))
} catch {
    FileHandle.standardError.write(Data("ERROR: \(error)\n".utf8))
    exit(1)
}
