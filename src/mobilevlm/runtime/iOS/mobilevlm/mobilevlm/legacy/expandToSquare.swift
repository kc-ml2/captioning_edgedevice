// //
// //  expandtosquare.swift
// //  mobilevlm
// //
// //  Created by hyeongseob jo on 4/18/26.
// //

// import Foundation

// func expandToSquare(_ tensor: ImageTensor, bgColor: [Float32]) -> ImageTensor {

//     let w = tensor.width
//     let h = tensor.height
//     let size = max(w, h)
    
//     // (HWC, RGB)
//     var result = [Float32](repeating: 0, count: size * size * 3)
    
//     // 1. background 채우기 (Python: np.ones * bg_color)
//     for i in stride(from: 0, to: result.count, by: 3) {
//         result[i]     = bgColor[0]
//         result[i + 1] = bgColor[1]
//         result[i + 2] = bgColor[2]
//     }
    
//     // 2. offset 계산 (Python과 동일)
//     let xOffset: Int
//     let yOffset: Int
    
//     if w > h {
//         xOffset = 0
//         yOffset = (w - h) / 2
//     } else {
//         xOffset = (h - w) / 2
//         yOffset = 0
//     }
    
//     // 3. 원본 복사
//     for y in 0..<h {
//         for x in 0..<w {
//             let srcIdx = (y * w + x) * 3
            
//             let dstX = x + xOffset
//             let dstY = y + yOffset
//             let dstIdx = (dstY * size + dstX) * 3
            
//             result[dstIdx]     = tensor.data[srcIdx]
//             result[dstIdx + 1] = tensor.data[srcIdx + 1]
//             result[dstIdx + 2] = tensor.data[srcIdx + 2]
//         }
//     }
    
//     return ImageTensor(data: result, width: size, height: size)
// }
