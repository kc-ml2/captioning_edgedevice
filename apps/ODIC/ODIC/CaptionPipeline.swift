import CoreGraphics
import Foundation
import MLX
import MLXNN
import SentencepieceTokenizer

actor CaptionPipeline {
    enum PipelineError: LocalizedError {
        case modelNotInstalled
        case invalidFrame
        case missingWeight(String)
        case unsupportedModel(String)

        var errorDescription: String? {
            switch self {
            case .modelNotInstalled:
                "MobileVLM 모델을 찾지 못했어요."
            case .invalidFrame:
                "카메라 프레임을 읽지 못했어요."
            case .missingWeight(let name):
                "모델 가중치가 없어요: \(name)"
            case .unsupportedModel(let reason):
                "지원하지 않는 모델이에요: \(reason)"
            }
        }
    }

    private struct Cache {
        var key: MLXArray
        var value: MLXArray
    }

    private let hiddenSize = 2048
    private let heads = 16
    private let headDimension = 128
    private let languageLayers = 24
    private let imageToken = -200
    private let quantizationGroupSize = 64
    private let quantizationBits = 4

    private var weights: [String: MLXArray]?
    private var tokenizer: SentencepieceTokenizer?

    func caption(
        image: CGImage,
        question: String = "What objects are visible in the scene?",
        progress: (@Sendable (String) async -> Void)? = nil
    ) async throws -> String {
        guard image.width > 0, image.height > 0 else {
            throw PipelineError.invalidFrame
        }

        try await loadIfNeeded(progress: progress)
        guard let weights, let tokenizer else {
            throw PipelineError.modelNotInstalled
        }

        await progress?("이미지를 살펴보고 있어요…")
        let ids = try tokenIDs(tokenizer, question: question)
        guard let imagePosition = ids.firstIndex(of: imageToken) else {
            throw PipelineError.invalidFrame
        }

        let pixels = loadPixels(image)
        let imageFeatures = try projector(vision(pixels, weights), weights)
        eval(imageFeatures)

        let before = try embed(Array(ids[..<imagePosition]), weights)
        let after = try embed(Array(ids[(imagePosition + 1)...]), weights)
        var hidden = concatenated([before, imageFeatures[0], after], axis: 0)
            .expandedDimensions(axis: 0)
        var caches: [Cache] = []

        await progress?("장면을 이해하고 있어요…")
        let prefillLength = hidden.shape[1]
        for layer in 0..<languageLayers {
            let result = try languageLayer(
                hidden,
                weights,
                layer,
                positions: arange(prefillLength),
                past: nil
            )
            hidden = result.0
            caches.append(result.1)
            eval(hidden)
        }

        await progress?("설명을 만들고 있어요…")
        var generated: [Int] = []
        for _ in 0..<40 {
            let normalized = rmsNorm(
                hidden,
                try required(weights, "language_model.norm.weight")
            )
            let logits = try linear(normalized, weights, "lm_head.weight")[0, hidden.shape[1] - 1]
            let token = Int(logits.argMax().item(Int32.self))
            generated.append(token)
            if token == 2 { break }

            hidden = try embed([token], weights).reshaped(1, 1, hiddenSize)
            for layer in 0..<languageLayers {
                let result = try languageLayer(
                    hidden,
                    weights,
                    layer,
                    positions: MLXArray([caches[layer].key.shape[2]]),
                    past: caches[layer]
                )
                hidden = result.0
                caches[layer] = result.1
                eval(hidden)
            }
        }

        return try tokenizer.decode(generated.map { $0 + 1 })
            .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func loadIfNeeded(
        progress: (@Sendable (String) async -> Void)?
    ) async throws {
        guard weights == nil || tokenizer == nil else { return }
        guard let directory = modelDirectory else {
            throw PipelineError.modelNotInstalled
        }

        await progress?("모델을 불러오고 있어요…")
        Memory.cacheLimit = 20 * 1024 * 1024
        try validateModelConfiguration(directory)
        let indexURL = directory.appending(path: "model.safetensors.index.json")
        let data = try Data(contentsOf: indexURL)
        let object = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let weightMap = object["weight_map"] as! [String: String]
        var loaded: [String: MLXArray] = [:]

        for shard in Set(weightMap.values).sorted() {
            loaded.merge(try loadArrays(url: directory.appending(path: shard))) { _, new in new }
        }

        weights = loaded
        tokenizer = try SentencepieceTokenizer(
            modelPath: directory.appending(path: "tokenizer.model").path
        )
    }

    private var modelDirectory: URL? {
        let fileManager = FileManager.default

        if let override = ProcessInfo.processInfo.environment["ODIC_MODEL_PATH"] {
            let url = URL(filePath: override, directoryHint: .isDirectory)
            if isValidModel(at: url) { return url }
        }

        if let support = try? fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        ) {
            let installed = support.appending(path: "MobileVLM", directoryHint: .isDirectory)
            if isValidModel(at: installed) { return installed }
        }

        #if DEBUG && targetEnvironment(simulator)
        var repository = URL(filePath: #filePath)
        for _ in 0..<4 { repository.deleteLastPathComponent() }
        let developmentModel = repository.appending(
            path: "workspace/mobilevlm-mlx/converted-fp16",
            directoryHint: .isDirectory
        )
        if isValidModel(at: developmentModel) { return developmentModel }
        #endif

        return nil
    }

    private func isValidModel(at directory: URL) -> Bool {
        let fileManager = FileManager.default
        let indexURL = directory.appending(path: "model.safetensors.index.json")
        let required = ["config.json", "model.safetensors.index.json", "tokenizer.model"]
        guard required.allSatisfy({ fileManager.fileExists(atPath: directory.appending(path: $0).path) }),
              let data = try? Data(contentsOf: indexURL),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let weightMap = object["weight_map"] as? [String: String] else {
            return false
        }
        return !weightMap.isEmpty && Set(weightMap.values).allSatisfy {
            fileManager.fileExists(atPath: directory.appending(path: $0).path)
        }
    }

    private func validateModelConfiguration(_ directory: URL) throws {
        let data = try Data(contentsOf: directory.appending(path: "config.json"))
        guard let config = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              config["weight_dtype"] as? String == "mixed_fp16_q4",
              let quantization = config["quantization"] as? [String: Any],
              quantization["bits"] as? Int == quantizationBits,
              quantization["group_size"] as? Int == quantizationGroupSize,
              quantization["mode"] as? String == "affine" else {
            throw PipelineError.unsupportedModel("iPhone에서는 MLX 4-bit 모델이 필요합니다.")
        }
    }

    private func required(_ weights: [String: MLXArray], _ name: String) throws -> MLXArray {
        guard let value = weights[name] else {
            throw PipelineError.missingWeight(name)
        }
        return value
    }

    private func linear(
        _ input: MLXArray,
        _ weights: [String: MLXArray],
        _ name: String
    ) throws -> MLXArray {
        let weight = try required(weights, name)
        let prefix = String(name.dropLast(".weight".count))
        if let scales = weights["\(prefix).scales"] {
            return quantizedMM(
                input,
                weight,
                scales: scales,
                biases: weights["\(prefix).biases"],
                transpose: true,
                groupSize: quantizationGroupSize,
                bits: quantizationBits,
                mode: .affine
            )
        }
        return input.matmul(weight.transposed())
    }

    private func embed(_ tokenIDs: [Int], _ weights: [String: MLXArray]) throws -> MLXArray {
        let name = "language_model.embed_tokens.weight"
        let weight = try required(weights, name)
        let indices = MLXArray(tokenIDs)
        guard let scales = weights["language_model.embed_tokens.scales"] else {
            return weight.take(indices, axis: 0)
        }
        let selectedWeight = weight.take(indices, axis: 0)
        let selectedScales = scales.take(indices, axis: 0)
        let selectedBiases = weights["language_model.embed_tokens.biases"]?.take(indices, axis: 0)
        return dequantized(
            selectedWeight,
            scales: selectedScales,
            biases: selectedBiases,
            groupSize: quantizationGroupSize,
            bits: quantizationBits,
            mode: .affine
        )
    }

    private func affine(
        _ input: MLXArray,
        _ weights: [String: MLXArray],
        _ prefix: String
    ) throws -> MLXArray {
        try linear(input, weights, "\(prefix).weight")
            + required(weights, "\(prefix).bias")
    }

    private func rmsNorm(_ input: MLXArray, _ weight: MLXArray) -> MLXArray {
        input * rsqrt((input * input).mean(axis: -1, keepDims: true) + 1e-6) * weight
    }

    private func layerNorm(
        _ input: MLXArray,
        _ weights: [String: MLXArray],
        _ prefix: String
    ) throws -> MLXArray {
        let mean = input.mean(axis: -1, keepDims: true)
        let centered = input - mean
        return centered
            * rsqrt((centered * centered).mean(axis: -1, keepDims: true) + 1e-5)
            * (try required(weights, "\(prefix).weight"))
            + (try required(weights, "\(prefix).bias"))
    }

    private func loadPixels(_ image: CGImage) -> MLXArray {
        let width = image.width
        let height = image.height
        let square = max(width, height)
        let target = 336
        let mean: [Float] = [0.48145466, 0.4578275, 0.40821073]
        let standardDeviation: [Float] = [0.26862954, 0.26130258, 0.27577711]
        let background = [UInt8(mean[0] * 255), UInt8(mean[1] * 255), UInt8(mean[2] * 255)]

        var squarePixels = [UInt8](repeating: 0, count: square * square * 4)
        for index in 0..<(square * square) {
            squarePixels[index * 4] = background[0]
            squarePixels[index * 4 + 1] = background[1]
            squarePixels[index * 4 + 2] = background[2]
            squarePixels[index * 4 + 3] = 255
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(
            data: &squarePixels,
            width: square,
            height: square,
            bitsPerComponent: 8,
            bytesPerRow: square * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!
        context.interpolationQuality = .none
        context.draw(
            image,
            in: CGRect(x: (square - width) / 2, y: (square - height) / 2, width: width, height: height)
        )

        var resized = [UInt8](repeating: 0, count: target * target * 4)
        let output = CGContext(
            data: &resized,
            width: target,
            height: target,
            bitsPerComponent: 8,
            bytesPerRow: target * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!
        output.interpolationQuality = .high
        output.draw(context.makeImage()!, in: CGRect(x: 0, y: 0, width: target, height: target))

        var values = [Float](repeating: 0, count: target * target * 3)
        for y in 0..<target {
            for x in 0..<target {
                let pixel = (y * target + x) * 4
                let outputIndex = (y * target + x) * 3
                values[outputIndex] = (Float(resized[pixel]) / 255 - mean[0]) / standardDeviation[0]
                values[outputIndex + 1] = (Float(resized[pixel + 1]) / 255 - mean[1]) / standardDeviation[1]
                values[outputIndex + 2] = (Float(resized[pixel + 2]) / 255 - mean[2]) / standardDeviation[2]
            }
        }
        return MLXArray(values, [1, target, target, 3])
    }

    private func vision(
        _ pixels: MLXArray,
        _ weights: [String: MLXArray]
    ) throws -> MLXArray {
        let root = "vision_model.vision_model"
        let patches = conv2d(
            pixels,
            try required(weights, "\(root).embeddings.patch_embedding.weight"),
            stride: 14
        ).reshaped(1, 576, 1024)
        let classification = try required(weights, "\(root).embeddings.class_embedding")
            .reshaped(1, 1, 1024)
        var hidden = concatenated([classification, patches], axis: 1)
            + (try required(weights, "\(root).embeddings.position_embedding.weight"))
        hidden = try layerNorm(hidden, weights, "\(root).pre_layrnorm")

        for layer in 0..<23 {
            let prefix = "\(root).encoder.layers.\(layer)"
            let normalized = try layerNorm(hidden, weights, "\(prefix).layer_norm1")
            let query = try affine(normalized, weights, "\(prefix).self_attn.q_proj")
                .reshaped(1, 577, 16, 64).transposed(0, 2, 1, 3)
            let key = try affine(normalized, weights, "\(prefix).self_attn.k_proj")
                .reshaped(1, 577, 16, 64).transposed(0, 2, 1, 3)
            let value = try affine(normalized, weights, "\(prefix).self_attn.v_proj")
                .reshaped(1, 577, 16, 64).transposed(0, 2, 1, 3)
            var attention = softmax(
                query.matmul(key.transposed(0, 1, 3, 2)) / 8,
                axis: -1
            ).matmul(value).transposed(0, 2, 1, 3).reshaped(1, 577, 1024)
            attention = try affine(attention, weights, "\(prefix).self_attn.out_proj")
            hidden = hidden + attention

            var mlp = try affine(
                layerNorm(hidden, weights, "\(prefix).layer_norm2"),
                weights,
                "\(prefix).mlp.fc1"
            )
            mlp = mlp * sigmoid(1.702 * mlp)
            hidden = hidden + (try affine(mlp, weights, "\(prefix).mlp.fc2"))
            eval(hidden)
        }
        return hidden[0..., 1..., 0...]
    }

    private func projector(
        _ input: MLXArray,
        _ weights: [String: MLXArray]
    ) throws -> MLXArray {
        var hidden = try affine(input, weights, "projector.mlp.mlp.0")
        hidden = gelu(hidden)
        hidden = try affine(hidden, weights, "projector.mlp.mlp.2")
        let pooled = AvgPool2d(kernelSize: 2, stride: 2)(hidden.reshaped(1, 24, 24, 2048))
        return (
            pooled
                + conv2d(
                    pooled,
                    try required(weights, "projector.peg.peg.0.weight"),
                    padding: 1,
                    groups: 2048
                )
                + (try required(weights, "projector.peg.peg.0.bias"))
        ).reshaped(1, 144, 2048)
    }

    private func rotateHalf(_ input: MLXArray) -> MLXArray {
        concatenated([
            -input[0..., 0..., 0..., (headDimension / 2)...],
            input[0..., 0..., 0..., ..<(headDimension / 2)],
        ], axis: -1)
    }

    private func rope(_ input: MLXArray, positions: MLXArray) -> MLXArray {
        let dimensions = arange(0, headDimension, step: 2).asType(.float32)
        let inverseFrequency = 1.0 / pow(10_000.0, dimensions / Float(headDimension))
        let frequencies = outer(positions.asType(.float32), inverseFrequency)
        let embedding = concatenated([frequencies, frequencies], axis: -1)
            .reshaped(1, 1, positions.size, headDimension)
        return input * cos(embedding) + rotateHalf(input) * sin(embedding)
    }

    private func languageLayer(
        _ input: MLXArray,
        _ weights: [String: MLXArray],
        _ layer: Int,
        positions: MLXArray,
        past: Cache?
    ) throws -> (MLXArray, Cache) {
        let prefix = "language_model.layers.\(layer)"
        let length = input.shape[1]
        let normalized = rmsNorm(
            input,
            try required(weights, "\(prefix).input_layernorm.weight")
        )
        let query = rope(
            try linear(normalized, weights, "\(prefix).self_attn.q_proj.weight")
                .reshaped(1, length, heads, headDimension).transposed(0, 2, 1, 3),
            positions: positions
        )
        let newKey = rope(
            try linear(normalized, weights, "\(prefix).self_attn.k_proj.weight")
                .reshaped(1, length, heads, headDimension).transposed(0, 2, 1, 3),
            positions: positions
        )
        let newValue = try linear(normalized, weights, "\(prefix).self_attn.v_proj.weight")
            .reshaped(1, length, heads, headDimension).transposed(0, 2, 1, 3)
        let key = past.map { concatenated([$0.key, newKey], axis: 2) } ?? newKey
        let value = past.map { concatenated([$0.value, newValue], axis: 2) } ?? newValue

        var scores = query.matmul(key.transposed(0, 1, 3, 2)) / sqrt(Float(headDimension))
        if past == nil {
            scores = scores + (1.0 - tri(length, dtype: .float32)) * -1e9
        }
        var attention = softmax(scores, axis: -1).matmul(value)
            .transposed(0, 2, 1, 3).reshaped(1, length, hiddenSize)
        attention = try linear(attention, weights, "\(prefix).self_attn.o_proj.weight")
        let residual = input + attention
        let postAttention = rmsNorm(
            residual,
            try required(weights, "\(prefix).post_attention_layernorm.weight")
        )
        let gate = silu(try linear(postAttention, weights, "\(prefix).mlp.gate_proj.weight"))
        let up = try linear(postAttention, weights, "\(prefix).mlp.up_proj.weight")
        let output = residual + (try linear(gate * up, weights, "\(prefix).mlp.down_proj.weight"))
        return (output, Cache(key: key, value: value))
    }

    private func tokenIDs(
        _ tokenizer: SentencepieceTokenizer,
        question: String
    ) throws -> [Int] {
        let prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n\(question) ASSISTANT:"
        let chunks = prompt.components(separatedBy: "<image>")
        var ids: [Int] = []
        for (index, chunk) in chunks.enumerated() {
            let encoded = try tokenizer.encode(chunk).map { $0 - 1 }
            if index == 0 {
                ids.append(1)
                ids.append(contentsOf: encoded)
            } else {
                ids.append(imageToken)
                ids.append(contentsOf: encoded.filter { $0 != 1 })
            }
        }
        return ids
    }
}
