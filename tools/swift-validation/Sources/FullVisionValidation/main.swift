import Foundation
import MLX
import MLXNN

let selectedLayerCount = 23

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
    return centered * rsqrt(variance + 1e-5)
        * (try require(weights, "\(prefix).weight"))
        + (try require(weights, "\(prefix).bias"))
}

func encoderLayer(_ input: MLXArray, weights: [String: MLXArray], prefix: String) throws -> MLXArray {
    let norm1 = try layerNorm(input, weights: weights, prefix: "\(prefix).layer_norm1")
    let q = try linear(norm1, weights: weights, prefix: "\(prefix).self_attn.q_proj")
        .reshaped(1, 577, 16, 64).transposed(0, 2, 1, 3)
    let k = try linear(norm1, weights: weights, prefix: "\(prefix).self_attn.k_proj")
        .reshaped(1, 577, 16, 64).transposed(0, 2, 1, 3)
    let v = try linear(norm1, weights: weights, prefix: "\(prefix).self_attn.v_proj")
        .reshaped(1, 577, 16, 64).transposed(0, 2, 1, 3)
    let probabilities = softmax(q.matmul(k.transposed(0, 1, 3, 2)) / sqrt(Float(64)), axis: -1)
    var attention = probabilities.matmul(v).transposed(0, 2, 1, 3).reshaped(1, 577, 1024)
    attention = try linear(attention, weights: weights, prefix: "\(prefix).self_attn.out_proj")
    let residual = input + attention

    let norm2 = try layerNorm(residual, weights: weights, prefix: "\(prefix).layer_norm2")
    var hidden = try linear(norm2, weights: weights, prefix: "\(prefix).mlp.fc1")
    hidden = hidden * sigmoid(1.702 * hidden)
    return residual + (try linear(hidden, weights: weights, prefix: "\(prefix).mlp.fc2"))
}

func run(fixtureURL: URL) throws {
    let weights = try loadArrays(url: fixtureURL)
    let root = "vision_model.vision_model"
    let pixels = try require(weights, "input.pixels_nhwc")
    let patches = conv2d(
        pixels,
        try require(weights, "\(root).embeddings.patch_embedding.weight"),
        stride: 14
    ).reshaped(1, 576, 1024)
    let cls = try require(weights, "\(root).embeddings.class_embedding").reshaped(1, 1, 1024)
    var hidden = concatenated([cls, patches], axis: 1)
        + (try require(weights, "\(root).embeddings.position_embedding.weight"))
    hidden = try layerNorm(hidden, weights: weights, prefix: "\(root).pre_layrnorm")

    for layer in 0 ..< selectedLayerCount {
        hidden = try encoderLayer(
            hidden, weights: weights, prefix: "\(root).encoder.layers.\(layer)"
        )
        eval(hidden)
        print("MLX vision layer \(layer + 1)/\(selectedLayerCount)")
    }

    let selected = hidden[0..., 1..., 0...]
    let expected = try require(weights, "expected.selected_features")
    let difference = abs(selected.asType(.float32) - expected.asType(.float32))
    let maxError = difference.max()
    let meanError = difference.mean()
    let cosineNumerator = (selected.asType(.float32) * expected.asType(.float32)).sum()
    let cosineDenominator = sqrt((selected.square().sum()) * (expected.square().sum()))
    let cosine = cosineNumerator / cosineDenominator
    let matches = allClose(selected, expected, rtol: 2e-2, atol: 2e-2)
    eval(maxError, meanError, cosine, matches)

    print("selected_features: passed=\(matches.item(Bool.self))")
    print("max_abs=\(maxError.item(Float.self))")
    print("mean_abs=\(meanError.item(Float.self))")
    print("cosine_similarity=\(cosine.item(Float.self))")
    print("shape=\(selected.shape)")

    guard matches.item(Bool.self) else {
        throw ValidationFailure(description: "Full MLX Swift CLIP output differs from PyTorch")
    }
    print("PASS: MLX Swift CLIP hidden_states[-2][:, 1:] matches PyTorch.")

    var projected = try linear(selected, weights: weights, prefix: "projector.mlp.mlp.0")
    projected = gelu(projected)
    projected = try linear(projected, weights: weights, prefix: "projector.mlp.mlp.2")
    let pooled = AvgPool2d(kernelSize: 2, stride: 2)(projected.reshaped(1, 24, 24, 2048))
    let positional = conv2d(
        pooled,
        try require(weights, "projector.peg.peg.0.weight"),
        padding: 1,
        groups: 2048
    ) + (try require(weights, "projector.peg.peg.0.bias"))
    let projectorOutput = (pooled + positional).reshaped(1, 144, 2048)
    let expectedProjector = try require(weights, "expected.projector_output")
    let projectorDifference = abs(
        projectorOutput.asType(.float32) - expectedProjector.asType(.float32)
    )
    let projectorMaxError = projectorDifference.max()
    let projectorMeanError = projectorDifference.mean()
    let projectorMatches = allClose(
        projectorOutput, expectedProjector, rtol: 3e-2, atol: 3e-2
    )
    eval(projectorMaxError, projectorMeanError, projectorMatches)
    print("projector_output: passed=\(projectorMatches.item(Bool.self))")
    print("projector_max_abs=\(projectorMaxError.item(Float.self))")
    print("projector_mean_abs=\(projectorMeanError.item(Float.self))")
    print("projector_shape=\(projectorOutput.shape)")
    guard projectorMatches.item(Bool.self) else {
        throw ValidationFailure(description: "Vision-to-projector output differs from PyTorch")
    }
    print("PASS: MLX Swift Vision Encoder -> LDPNetV2 matches PyTorch end to end.")
}

do {
    guard CommandLine.arguments.count == 2 else {
        throw ValidationFailure(
            description: "Usage: FullVisionValidation <full-vision-fixture.safetensors>"
        )
    }
    try run(fixtureURL: URL(filePath: CommandLine.arguments[1]))
} catch {
    FileHandle.standardError.write(Data("ERROR: \(error)\n".utf8))
    exit(1)
}
