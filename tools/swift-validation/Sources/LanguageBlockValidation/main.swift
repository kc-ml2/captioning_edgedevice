import Foundation
import MLX
import MLXNN

let sequenceLength = 32
let hiddenSize = 2048
let headCount = 16
let headDimension = 128

struct ValidationFailure: Error, CustomStringConvertible {
    let description: String
}

func require(_ arrays: [String: MLXArray], _ name: String) throws -> MLXArray {
    guard let value = arrays[name] else {
        throw ValidationFailure(description: "Missing fixture tensor: \(name)")
    }
    return value
}

func linear(_ input: MLXArray, weights: [String: MLXArray], name: String) throws -> MLXArray {
    input.matmul(try require(weights, name).transposed())
}

func rmsNorm(_ input: MLXArray, weight: MLXArray) -> MLXArray {
    input * rsqrt((input * input).mean(axis: -1, keepDims: true) + 1e-6) * weight
}

func rotateHalf(_ input: MLXArray) -> MLXArray {
    let first = input[0..., 0..., 0..., ..<(headDimension / 2)]
    let second = input[0..., 0..., 0..., (headDimension / 2)...]
    return concatenated([-second, first], axis: -1)
}

func applyRoPE(_ q: MLXArray, _ k: MLXArray) -> (MLXArray, MLXArray) {
    let positions = arange(sequenceLength).asType(.float32)
    let dimensions = arange(0, headDimension, step: 2).asType(.float32)
    let inverseFrequency = 1.0 / pow(10_000.0, dimensions / Float(headDimension))
    let frequencies = outer(positions, inverseFrequency)
    let embedding = concatenated([frequencies, frequencies], axis: -1)
        .reshaped(1, 1, sequenceLength, headDimension)
    let cosine = cos(embedding)
    let sine = sin(embedding)
    return (q * cosine + rotateHalf(q) * sine, k * cosine + rotateHalf(k) * sine)
}

func report(_ name: String, actual: MLXArray, expected: MLXArray, atol: Double, rtol: Double) -> Bool {
    let difference = abs(actual.asType(.float32) - expected.asType(.float32))
    let maxError = difference.max()
    let meanError = difference.mean()
    let matches = allClose(actual, expected, rtol: rtol, atol: atol)
    eval(maxError, meanError, matches)
    print(
        "\(name): passed=\(matches.item(Bool.self)) max_abs=\(maxError.item(Float.self)) "
            + "mean_abs=\(meanError.item(Float.self)) shape=\(actual.shape)"
    )
    return matches.item(Bool.self)
}

func run(fixtureURL: URL) throws {
    let weights = try loadArrays(url: fixtureURL)
    let layer = "language_model.layers.0"

    // The fixture stores the exact embedding rows in token order to avoid
    // duplicating the 125 MiB embedding table in this component test.
    let embeddings = try require(weights, "input.embedding_rows").reshaped(1, sequenceLength, hiddenSize)
    let norm1 = rmsNorm(embeddings, weight: try require(weights, "\(layer).input_layernorm.weight"))
    var q = try linear(norm1, weights: weights, name: "\(layer).self_attn.q_proj.weight")
    var k = try linear(norm1, weights: weights, name: "\(layer).self_attn.k_proj.weight")
    let v = try linear(norm1, weights: weights, name: "\(layer).self_attn.v_proj.weight")
        .reshaped(1, sequenceLength, headCount, headDimension).transposed(0, 2, 1, 3)
    q = q.reshaped(1, sequenceLength, headCount, headDimension).transposed(0, 2, 1, 3)
    k = k.reshaped(1, sequenceLength, headCount, headDimension).transposed(0, 2, 1, 3)
    let (qRope, kRope) = applyRoPE(q, k)

    var scores = qRope.matmul(kRope.transposed(0, 1, 3, 2)) / sqrt(Float(headDimension))
    let allowed = tri(sequenceLength, dtype: .float32)
    scores = scores + (1.0 - allowed) * -1e9
    let probabilities = softmax(scores, axis: -1)
    var attention = probabilities.matmul(v).transposed(0, 2, 1, 3)
        .reshaped(1, sequenceLength, hiddenSize)
    attention = try linear(attention, weights: weights, name: "\(layer).self_attn.o_proj.weight")
    let attentionResidual = embeddings + attention

    let norm2 = rmsNorm(
        attentionResidual,
        weight: try require(weights, "\(layer).post_attention_layernorm.weight")
    )
    let gate = silu(try linear(norm2, weights: weights, name: "\(layer).mlp.gate_proj.weight"))
    let up = try linear(norm2, weights: weights, name: "\(layer).mlp.up_proj.weight")
    let mlp = try linear(gate * up, weights: weights, name: "\(layer).mlp.down_proj.weight")
    let output = attentionResidual + mlp
    let finalNorm = rmsNorm(output, weight: try require(weights, "language_model.norm.weight"))
    let logits = try linear(finalNorm, weights: weights, name: "lm_head.weight")

    let checks = [
        report("embeddings", actual: embeddings, expected: try require(weights, "expected.embeddings"), atol: 0, rtol: 0),
        report("norm1", actual: norm1, expected: try require(weights, "expected.norm1"), atol: 5e-5, rtol: 5e-5),
        report("q", actual: q, expected: try require(weights, "expected.q"), atol: 2e-4, rtol: 2e-4),
        report("k", actual: k, expected: try require(weights, "expected.k"), atol: 2e-4, rtol: 2e-4),
        report("v", actual: v, expected: try require(weights, "expected.v"), atol: 2e-4, rtol: 2e-4),
        report("q.rope", actual: qRope, expected: try require(weights, "expected.q_rope"), atol: 3e-4, rtol: 3e-4),
        report("k.rope", actual: kRope, expected: try require(weights, "expected.k_rope"), atol: 3e-4, rtol: 3e-4),
        report("attention", actual: attention, expected: try require(weights, "expected.attention"), atol: 1e-3, rtol: 1e-3),
        report("attention.residual", actual: attentionResidual, expected: try require(weights, "expected.attention_residual"), atol: 1e-3, rtol: 1e-3),
        report("norm2", actual: norm2, expected: try require(weights, "expected.norm2"), atol: 1e-3, rtol: 1e-3),
        report("mlp.gate", actual: gate, expected: try require(weights, "expected.mlp_gate"), atol: 2e-3, rtol: 2e-3),
        report("mlp.up", actual: up, expected: try require(weights, "expected.mlp_up"), atol: 2e-3, rtol: 2e-3),
        report("block.output", actual: output, expected: try require(weights, "expected.output"), atol: 3e-3, rtol: 3e-3),
        report("final_norm", actual: finalNorm, expected: try require(weights, "expected.final_norm"), atol: 3e-3, rtol: 3e-3),
        report("logits", actual: logits, expected: try require(weights, "expected.logits"), atol: 1e-2, rtol: 1e-2),
    ]

    let lastTokenLogits = logits[0, sequenceLength - 1]
    let expectedLastTokenLogits = try require(weights, "expected.logits")[0, sequenceLength - 1]
    let token = lastTokenLogits.argMax().item(Int32.self)
    let expectedToken = expectedLastTokenLogits.argMax().item(Int32.self)
    print("last_token_argmax: actual=\(token) expected=\(expectedToken)")

    guard checks.allSatisfy({ $0 }) && token == expectedToken else {
        throw ValidationFailure(description: "MLX Swift MobileLlama block differs from PyTorch")
    }
    print("PASS: MLX Swift MobileLlama block 0 and LM head match PyTorch.")
}

do {
    guard CommandLine.arguments.count == 2 else {
        throw ValidationFailure(
            description: "Usage: LanguageBlockValidation <language-block-fixture.safetensors>"
        )
    }
    try run(fixtureURL: URL(filePath: CommandLine.arguments[1]))
} catch {
    FileHandle.standardError.write(Data("ERROR: \(error)\n".utf8))
    exit(1)
}
