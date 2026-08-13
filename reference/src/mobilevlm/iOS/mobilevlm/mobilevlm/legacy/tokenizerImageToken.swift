//import Foundation
//import SentencepieceTokenizer
//
//
//let BOS_ID = 1
//let IMAGE_TOKEN_INDEX = -200
//
//func tokenizeImagePrompt(prompt: String) -> [Int]? {
//    guard let path = Bundle.main.path(forResource: "tokenizer", ofType: "model") else {
//        print("❌ model not found")
//        return nil
//    }
//
//    do {
//        let tokenizer = try SentencepieceTokenizer(modelPath: path)
//        
//        let promptChunks: [[Int]] = try prompt
//            .components(separatedBy: "<image>")
//            .map { chunk in
//                let encoded = try tokenizer.encode(chunk).map { Int($0) - 1 }
//                return [BOS_ID] + encoded
//            }
//
//        func insertSeparator(_ X: [[Int]], sep: [Int]) -> [[Int]] {
//            var result: [[Int]] = []
//            for i in 0..<X.count {
//                result.append(X[i])
//                if i < X.count - 1 {
//                    result.append(sep)
//                }
//            }
//            return result
//        }
//
//        var inputIds: [Int] = []
//        var offset = 0
//
//        if let first = promptChunks.first,
//           let firstToken = first.first,
//           firstToken == BOS_ID {
//            offset = 1
//            inputIds.append(firstToken)
//        }
//
//        let sep = Array(repeating: IMAGE_TOKEN_INDEX, count: offset + 1)
//
//        for x in insertSeparator(promptChunks, sep: sep) {
//            inputIds.append(contentsOf: x.dropFirst(offset))
//        }
//
//        return inputIds
//
//    } catch {
//        print("❌ tokenizer error:", error)
//        return nil
//    }
//}
