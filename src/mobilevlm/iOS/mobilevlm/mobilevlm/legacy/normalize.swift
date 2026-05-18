// //
// //  normalizeToCHW.swift
// //  mobilevlm
// //
// //  Created by hyeongseob jo on 4/18/26.
// //

// import Foundation

// func normalize(_ tensor: ImageTensor) -> [Float32] {
    
//     let width = tensor.width
//     let height = tensor.height
//     let data = tensor.data   // HWC, Float32 (0~255)
    
//     let pixelCount = width * height
    
//     // CLIP 기준
//     let mean: [Float32] = [0.48145466, 0.4578275, 0.40821073]
//     let std:  [Float32] = [0.26862954, 0.26130258, 0.27577711]
    
//     // 결과 (CHW)
//     var output = [Float32](repeating: 0, count: 3 * pixelCount)
    
//     for i in 0..<pixelCount {
        
//         // HWC → RGB
//         let r = data[i*3 + 0] / 255.0
//         let g = data[i*3 + 1] / 255.0
//         let b = data[i*3 + 2] / 255.0
        
//         // normalize
//         let nr = (r - mean[0]) / std[0]
//         let ng = (g - mean[1]) / std[1]
//         let nb = (b - mean[2]) / std[2]
        
//         // CHW 저장
//         output[i]                 = nr                // R 채널
//         output[i + pixelCount]    = ng                // G 채널
//         output[i + pixelCount*2]  = nb                // B 채널
//     }
    
//     return output
// }
