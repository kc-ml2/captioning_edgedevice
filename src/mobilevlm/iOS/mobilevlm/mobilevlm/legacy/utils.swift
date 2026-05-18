// //
// //  utils.swift
// //  mobilevlm
// //
// //  Created by hyeongseob jo on 4/17/26.
// //

// import Foundation

// func saveToDocuments(_ array: [Float32], filename: String) {
    
//     // Documents 경로
//     let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
//         .appendingPathComponent(filename)
    
//     // Float32 → Data (binary)
//     let data = array.withUnsafeBufferPointer {
//         Data(buffer: $0)
//     }
    
//     do {
//         try data.write(to: url)
//         print("✅ Saved to:", url)
//     } catch {
//         print("❌ Save failed:", error)
//     }
// }
