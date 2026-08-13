//  MultimodalProcessor.swift


import Foundation
import UIKit
import CoreML

final class MultimodalProcessor {
    // =========================
    // Multimodal Processing
    // =========================

    func mobilevlm(
        uiImage: UIImage
    ) -> String? {
        
        let totalStart = CFAbsoluteTimeGetCurrent()
        
        // =========================
        // Vision Encoder
        // =========================

        let imageFeatures = encodeImage(uiImage)
        
        // =========================
        // Multimodal Embedding
        // =========================

        guard let multimodalEmbeddings = buildInputEmbeddings(
            imageFeatures: imageFeatures
        )
        else {
            return nil
        }
        
        // =========================
        // Large Language Model
        // =========================


        var generatedTokens: [Int] = []

        var curPos = multimodalEmbeddings.shape[1].intValue  // 194

        let kvCache = createKVCache(seqLen: 1)  // [48,16,1,128]
        
        let attentionMask = try! MLMultiArray(
            shape: [1, NSNumber(value: curPos + 1)],
            dataType: .float32
        )

        let maskPtr = attentionMask.dataPointer.bindMemory(
            to: Float32.self,
            capacity: attentionMask.count
        )
        
        maskPtr[0] = 0.0

        for i in 1..<attentionMask.count {
            maskPtr[i] = 1.0
        }
        
        // =========================
        // Prefill
        // =========================
                
        let currentKVBuffer =
            createKVCache(
                seqLen: 235
            )  // [48,16,235,128]

        let result: (
            hidden_states: MLMultiArray,
            presentKV: MLMultiArray
        )
        
        let prefillload = CFAbsoluteTimeGetCurrent()

        guard (try? LLMManager.shared.loadPrefill(
            modelURL: ModelManager.shared.llmModelURL
        )) != nil else {

            print("❌ Prefill model load failed")
            return nil
        }
        
        print(String(
            format: "⏱ Prefill Loading Time: %.3f sec",
            CFAbsoluteTimeGetCurrent() - prefillload
        ))
        
        let prefillStart = CFAbsoluteTimeGetCurrent()
        
        do {
            result = try LLMManager.shared.run(
                mode: .prefill,
                inputsEmbeds: multimodalEmbeddings,
                attentionMask: attentionMask,
                kvCache: kvCache
            )

        } catch {

            print("❌ Prefill failed: \(error)")
            return nil
        }

        print(String(
            format: "⏱ Prefill Time: %.3f sec",
            CFAbsoluteTimeGetCurrent() - prefillStart
        ))
        
        let logits = lmHead(
            hiddenStates: result.hidden_states
        )

        let nextToken = getNextToken(
            logits: logits
        )
        
        copyKVToBuffer(
            src: result.presentKV,
            dst: currentKVBuffer,
            length: curPos + 1
        )

        LLMManager.shared.unloadPrefill()
        print("Prefill unloaded")
        
        generatedTokens.append(nextToken)

        curPos += 1
        
        var nextEmbed = makeNextTokenEmbedding(
            tokenId: nextToken
        )

        // =========================
        // Decoder
        // =========================
        
        let decoderMask = try! MLMultiArray(
            shape: [1, NSNumber(value: 236)],
            dataType: .float32
        )

        let maskPtr2 = decoderMask.dataPointer.bindMemory(
            to: Float32.self,
            capacity: decoderMask.count
        )

        maskPtr2[0] = 0.0
        
        for i in 1..<235 {
            maskPtr2[i] = (i < curPos) ? 1.0 : 0.0
        }
        
        maskPtr2[235] = 1.0
        
        do {
            let maxTokens = 40
            let eosTokenId = 2
            
            let decoderload = CFAbsoluteTimeGetCurrent()
            
            try LLMManager.shared.loadDecode(
                modelURL: ModelManager.shared.llmModelURL
            )
            
            print(String(
                format: "⏱ Decoder Loading Time: %.3f sec",
                CFAbsoluteTimeGetCurrent() - decoderload
            ))

            defer {
                LLMManager.shared.unloadDecode()
                print("Decoder unloaded")
            }
            
            var shouldStop = false
            
            let decoderStart = CFAbsoluteTimeGetCurrent()

            for _ in 0..<maxTokens {
                
                autoreleasepool {
                    
                    do {
                        
                        let result =
                        try LLMManager.shared.run(
                            mode: .decode,
                            inputsEmbeds: nextEmbed,
                            attentionMask: decoderMask,
                            kvCache: currentKVBuffer
                        )
                        
                        let logits = lmHead(
                            hiddenStates: result.hidden_states
                        )
                        
                        let token = getNextToken(
                            logits: logits
                        )
                                                
                        appendLastKV(
                            src: result.presentKV,
                            dst: currentKVBuffer,
                            position: curPos
                        )
                        
                        nextEmbed = makeNextTokenEmbedding(
                            tokenId: token
                        )
                        
                        generatedTokens.append(token)
                        
                        if token == eosTokenId {
                            shouldStop = true
                        }
                        
                        maskPtr2[curPos] = 1.0
                        
                        curPos += 1

                    } catch {

                        shouldStop = true
                    }
                }

                if shouldStop {
                    break
                }
            }
            
            let decoderTime = CFAbsoluteTimeGetCurrent() - decoderStart
            let decoderTokens = generatedTokens.count
            
            print(String(
                format: "⏱ Decoder: %.3f sec | Tokens: %d | ⏱ Avg: %.3f sec/token",
                decoderTime,
                decoderTokens,
                decoderTime / Double(decoderTokens)
            ))

        } catch {

            print("❌ Decode failed: \(error)")
            return nil
        }
        
        // =========================
        // Decode Tokens
        // =========================

        guard let caption = decodeTokens(
            generatedTokens: generatedTokens
        ) else {
            return nil
        }
        
        print(String(
            format: "⏱ Total Time: %.3f sec",
            CFAbsoluteTimeGetCurrent() - totalStart
        ))

        print("caption: \(caption)")

        print("========================================")

        return caption
        
    }
}
