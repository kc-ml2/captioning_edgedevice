////  MultimodalProcessor.swift
//
//
//import Foundation
//import UIKit
//import CoreML
//
//final class MultimodalProcessor {
//    // =========================
//    // Multimodal Processing
//    // =========================
//
//    func mobilevlm(
//        uiImage: UIImage
//    ) -> String? {
//        
//        let totalStart = CFAbsoluteTimeGetCurrent()
//        
//        // =========================
//        // Vision Encoder
//        // =========================
//
//        let imageFeatures = encodeImage(uiImage)
//        
//        // =========================
//        // Multimodal Embedding
//        // =========================
//
//        let mmInputStart = CFAbsoluteTimeGetCurrent()
//
//        let question =
//            "What objects are visible in the scene?"
//
//        guard let multimodalEmbeddings =
//            buildInputEmbeddings(
//                question: question,
//                imageFeatures: imageFeatures
//            )
//        else {
//            return nil
//        }
//
//        print(String(
//            format: "⏱ Multimodal Input Time: %.3f sec",
//            CFAbsoluteTimeGetCurrent() - mmInputStart
//        ))
//
//        var generatedTokens: [Int] = []
//
//        var curPos = multimodalEmbeddings.shape[1].intValue  // 194
//
//        let kvCache = createKVCache(
//            seqLen: 2
//        )  // [48,16,1,128]
//        
//        let attentionMask = try! MLMultiArray(
//            shape: [1, NSNumber(value: curPos + 2)],
//            dataType: .float32
//        )
//
//        let maskPtr = attentionMask.dataPointer.bindMemory(
//            to: Float32.self,
//            capacity: attentionMask.count
//        )
//
//        for i in 2..<(curPos + 2) {
//            maskPtr[i] = 1.0
//        }
//        
//        // =========================
//        // Prefill
//        // =========================
//                
//        let currentKVBuffer =
//            createKVCache(
//                seqLen: 236
//            )  // [48,16,236,128]
//        
//        let prefillStart = CFAbsoluteTimeGetCurrent()
//
//        guard let result = runsingleLLM(
//            inputsEmbeds: multimodalEmbeddings,
//            attentionMask: attentionMask,
//            kvCache: kvCache
//        ) else {
//            return nil
//        }
//
//        print(String(
//            format: "⏱ Prefill Time: %.3f sec",
//            CFAbsoluteTimeGetCurrent() - prefillStart
//        ))
//        
//        let logits = lmHead(
//            hiddenStates: result.hidden_states
//        )
//
//        let nextToken = getNextToken(
//            logits: logits
//        )
//
//        print("token: ", nextToken)
//        
//        copyKVToBuffer(
//            src: result.presentKV,
//            dst: currentKVBuffer,   // [48,16,196,128]
//            length: curPos + 2
//        )
//        
//        generatedTokens.append(nextToken)
//
//        curPos += 2
//        
//        var nextEmbed = makeNextTokenEmbedding(
//            tokenId: nextToken
//        )
//        
//        let maxSeqLen = 238
//
//        let decoderMask = try! MLMultiArray(
//            shape: [1, NSNumber(value: maxSeqLen)],
//            dataType: .float32
//        )
//
//        let maskPtr2 = decoderMask.dataPointer.bindMemory(
//            to: Float32.self,
//            capacity: decoderMask.count
//        )
//
//        for i in 2..<curPos {
//            maskPtr2[i] = 1.0
//        }
//
//        maskPtr2[maxSeqLen - 1] = 1.0
//
//        let maxNewTokens = 40
//        let eosTokenId = 2
//
//        // =========================
//        // Decoder
//        // =========================
//        
//        let decoderStart = CFAbsoluteTimeGetCurrent()
//        
//        for _ in 0..<maxNewTokens {
//
//            var decodedToken: Int?
//
//            autoreleasepool {
//                
//                let maxEmbeddingInput = try! MLMultiArray(
//                    shape: [1, 2, 2048],
//                    dataType: .float32
//                )
//
//                let maxPtr = maxEmbeddingInput.dataPointer.bindMemory(
//                    to: Float32.self,
//                    capacity: maxEmbeddingInput.count
//                )
//
//                let curPtr = nextEmbed.dataPointer.bindMemory(
//                    to: Float32.self,
//                    capacity: nextEmbed.count
//                )
//
//                for i in 0..<2048 {
//                    maxPtr[2048 + i] = curPtr[i]
//                }
//
//                guard let result = runsingleLLM(
//                    inputsEmbeds: maxEmbeddingInput,
//                    attentionMask: decoderMask,
//                    kvCache: currentKVBuffer
//                ) else {
//                    return
//                }
//                
//                let logits = lmHead(
//                    hiddenStates: result.hidden_states
//                )
//
//                let token = getNextToken(
//                    logits: logits
//                )
//
//                print("token:", token)
//                
//                maskPtr2[curPos] = 1.0
//
//                appendLastKV(
//                    src: result.presentKV,
//                    dst: currentKVBuffer,
//                    position: curPos
//                )
//
//                nextEmbed = makeNextTokenEmbedding(
//                    tokenId: token
//                )
//
//                generatedTokens.append(token)
//
//                decodedToken = token
//            }
//
//            guard let token = decodedToken else {
//                print("Decoder failed")
//                break
//            }
//
//            if token == eosTokenId {
//                break
//            }
//
//            curPos += 1
//        }
//
//        print(String(
//            format: "⏱ Avg Decoder Time per Toekn: %.3f sec",
//            (CFAbsoluteTimeGetCurrent() - decoderStart) / Double(generatedTokens.count)
//        ))
//        
//        // =========================
//        // Decode Tokens
//        // =========================
//
//        guard let caption = decodeTokens(
//            generatedTokens: generatedTokens
//        ) else {
//            return nil
//        }
//
//        print("caption: \(caption)")
//        
//        print(String(
//            format: "⏱ Total Time: %.3f sec",
//            CFAbsoluteTimeGetCurrent() - totalStart
//        ))
//
//        print("========================================")
//
//        return caption
//        
//    }
//}
