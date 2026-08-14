import Foundation
import MLX
import MLXNN

let sequenceLength = 16
let hiddenSize = 2048
let headCount = 16
let headDimension = 128
let layerCount = 24

struct ValidationFailure: Error, CustomStringConvertible {
    let description: String
}

func require(_ arrays: [String: MLXArray], _ name: String) throws -> MLXArray {
    guard let value = arrays[name] else { throw ValidationFailure(description: "Missing: \(name)") }
    return value
}

func linear(_ input: MLXArray, _ weights: [String: MLXArray], _ name: String) throws -> MLXArray {
    input.matmul(try require(weights, name).transposed())
}

func rmsNorm(_ input: MLXArray, _ weight: MLXArray) -> MLXArray {
    input * rsqrt((input * input).mean(axis: -1, keepDims: true) + 1e-6) * weight
}

func rotateHalf(_ input: MLXArray) -> MLXArray {
    concatenated([
        -input[0..., 0..., 0..., (headDimension / 2)...],
        input[0..., 0..., 0..., ..<(headDimension / 2)],
    ], axis: -1)
}

func applyRoPE(_ q: MLXArray, _ k: MLXArray) -> (MLXArray, MLXArray) {
    let positions = arange(sequenceLength).asType(.float32)
    let dimensions = arange(0, headDimension, step: 2).asType(.float32)
    let inverseFrequency = 1.0 / pow(10_000.0, dimensions / Float(headDimension))
    let frequencies = outer(positions, inverseFrequency)
    let embedding = concatenated([frequencies, frequencies], axis: -1)
        .reshaped(1, 1, sequenceLength, headDimension)
    return (
        q * cos(embedding) + rotateHalf(q) * sin(embedding),
        k * cos(embedding) + rotateHalf(k) * sin(embedding)
    )
}

func decoderLayer(_ input: MLXArray, _ weights: [String: MLXArray], layer: Int) throws -> MLXArray {
    let prefix = "language_model.layers.\(layer)"
    let norm1 = rmsNorm(input, try require(weights, "\(prefix).input_layernorm.weight"))
    var q = try linear(norm1, weights, "\(prefix).self_attn.q_proj.weight")
        .reshaped(1, sequenceLength, headCount, headDimension).transposed(0, 2, 1, 3)
    var k = try linear(norm1, weights, "\(prefix).self_attn.k_proj.weight")
        .reshaped(1, sequenceLength, headCount, headDimension).transposed(0, 2, 1, 3)
    let v = try linear(norm1, weights, "\(prefix).self_attn.v_proj.weight")
        .reshaped(1, sequenceLength, headCount, headDimension).transposed(0, 2, 1, 3)
    (q, k) = applyRoPE(q, k)
    var scores = q.matmul(k.transposed(0, 1, 3, 2)) / sqrt(Float(headDimension))
    scores = scores + (1.0 - tri(sequenceLength, dtype: .float32)) * -1e9
    var attention = softmax(scores, axis: -1).matmul(v).transposed(0, 2, 1, 3)
        .reshaped(1, sequenceLength, hiddenSize)
    attention = try linear(attention, weights, "\(prefix).self_attn.o_proj.weight")
    let residual = input + attention
    let norm2 = rmsNorm(residual, try require(weights, "\(prefix).post_attention_layernorm.weight"))
    let gate = silu(try linear(norm2, weights, "\(prefix).mlp.gate_proj.weight"))
    let up = try linear(norm2, weights, "\(prefix).mlp.up_proj.weight")
    return residual + (try linear(gate * up, weights, "\(prefix).mlp.down_proj.weight"))
}

func run(_ url: URL) throws {
    let weights = try loadArrays(url: url)
    var hidden = try require(weights, "input.embedding_rows").reshaped(1, sequenceLength, hiddenSize)
    for layer in 0 ..< layerCount {
        hidden = try decoderLayer(hidden, weights, layer: layer)
        eval(hidden)
        print("MLX language layer \(layer + 1)/\(layerCount)")
    }
    let finalNorm = rmsNorm(hidden, try require(weights, "language_model.norm.weight"))
    let logits = try linear(finalNorm, weights, "lm_head.weight")

    func compare(_ name: String, _ actual: MLXArray, _ expected: MLXArray, tolerance: Double) -> Bool {
        let difference = abs(actual.asType(.float32) - expected.asType(.float32))
        let maxError = difference.max()
        let meanError = difference.mean()
        let cosine = (actual.asType(.float32) * expected.asType(.float32)).sum()
            / sqrt(actual.square().sum() * expected.square().sum())
        let matches = allClose(actual, expected, rtol: tolerance, atol: tolerance)
        eval(maxError, meanError, cosine, matches)
        print("\(name): passed=\(matches.item(Bool.self)) max_abs=\(maxError.item(Float.self)) mean_abs=\(meanError.item(Float.self)) cosine=\(cosine.item(Float.self))")
        return matches.item(Bool.self)
    }

    let hiddenPass = compare("hidden", hidden, try require(weights, "expected.hidden"), tolerance: 3e-2)
    let normPass = compare("final_norm", finalNorm, try require(weights, "expected.final_norm"), tolerance: 3e-2)
    let logitsPass = compare("logits", logits, try require(weights, "expected.logits"), tolerance: 5e-2)
    let actualToken = logits[0, sequenceLength - 1].argMax().item(Int32.self)
    let expectedToken = try require(weights, "expected.logits")[0, sequenceLength - 1].argMax().item(Int32.self)
    print("last_token_argmax: actual=\(actualToken) expected=\(expectedToken)")

    guard hiddenPass && normPass && logitsPass && actualToken == expectedToken else {
        throw ValidationFailure(description: "Full MobileLlama prefill differs from PyTorch")
    }
    print("PASS: MLX Swift all 24 MobileLlama layers and prefill logits match PyTorch.")
}

do {
    guard CommandLine.arguments.count == 2 else {
        throw ValidationFailure(description: "Usage: FullLanguageValidation <fixture.safetensors>")
    }
    try run(URL(filePath: CommandLine.arguments[1]))
} catch {
    FileHandle.standardError.write(Data("ERROR: \(error)\n".utf8))
    exit(1)
}
