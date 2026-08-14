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

func linear(_ input: MLXArray, weight: MLXArray, bias: MLXArray) -> MLXArray {
    input.matmul(weight.transposed()) + bias
}

func report(_ name: String, actual: MLXArray, expected: MLXArray, atol: Double, rtol: Double) -> Bool {
    let difference = abs(actual.asType(.float32) - expected.asType(.float32))
    let maxAbsoluteError = difference.max()
    let meanAbsoluteError = difference.mean()
    let matches = allClose(actual, expected, rtol: rtol, atol: atol)
    eval(maxAbsoluteError, meanAbsoluteError, matches)

    let passed = matches.item(Bool.self)
    let maxError = maxAbsoluteError.item(Float.self)
    let meanError = meanAbsoluteError.item(Float.self)
    print("\(name): passed=\(passed) max_abs=\(maxError) mean_abs=\(meanError) shape=\(actual.shape)")
    return passed
}

func run(fixtureURL: URL) throws {
    let fixture = try loadArrays(url: fixtureURL)

    let input = try require(fixture, "input")
    let mlp0 = linear(
        input,
        weight: try require(fixture, "projector.mlp.mlp.0.weight"),
        bias: try require(fixture, "projector.mlp.mlp.0.bias")
    )
    let activated = gelu(mlp0)
    let mlp2 = linear(
        activated,
        weight: try require(fixture, "projector.mlp.mlp.2.weight"),
        bias: try require(fixture, "projector.mlp.mlp.2.bias")
    )

    // PyTorch adaptive_avg_pool2d from 24x24 to 12x12 is exactly a 2x2,
    // stride-2 average pool. MLX convolution and pooling use NHWC.
    let image = mlp2.reshaped(1, 24, 24, 2048)
    let pooled = AvgPool2d(kernelSize: 2, stride: 2)(image)
    let positional = conv2d(
        pooled,
        try require(fixture, "projector.peg.peg.0.weight"),
        padding: 1,
        groups: 2048
    ) + (try require(fixture, "projector.peg.peg.0.bias"))
    let output = (pooled + positional).reshaped(1, 144, 2048)

    let checks = [
        report("mlp0", actual: mlp0, expected: try require(fixture, "expected.mlp0"), atol: 2e-3, rtol: 2e-3),
        report("gelu", actual: activated, expected: try require(fixture, "expected.gelu"), atol: 2e-3, rtol: 2e-3),
        report("mlp2", actual: mlp2, expected: try require(fixture, "expected.mlp2"), atol: 4e-3, rtol: 4e-3),
        report("pool", actual: pooled, expected: try require(fixture, "expected.pooled_nhwc"), atol: 4e-3, rtol: 4e-3),
        report("projector.output", actual: output, expected: try require(fixture, "expected.output"), atol: 1e-2, rtol: 1e-2),
    ]

    guard checks.allSatisfy({ $0 }) else {
        throw ValidationFailure(description: "MLX Swift projector differs from PyTorch reference")
    }
    print("PASS: MLX Swift LDPNetV2 projector matches the PyTorch reference.")
}

do {
    guard CommandLine.arguments.count == 2 else {
        throw ValidationFailure(
            description: "Usage: swift run ProjectorValidation <projector-fixture.safetensors>"
        )
    }
    try run(fixtureURL: URL(filePath: CommandLine.arguments[1]))
} catch {
    FileHandle.standardError.write(Data("ERROR: \(error)\n".utf8))
    exit(1)
}
