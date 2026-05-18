// //
// //  resizeBicubic.swift
// //  mobilevlm
// //
// //  Created by hyeongseob jo on 4/18/26.
// //

// import Foundation
// import CoreGraphics


// func resizeBicubic(_ tensor: ImageTensor, targetSize: Int) -> ImageTensor? {
    
//     let width = tensor.width
//     let height = tensor.height
//     let data = tensor.data
    
//     // Float32 → UInt8
//     var uint8Data = [UInt8](repeating: 0, count: width * height * 4)
    
//     for i in 0..<(width * height) {
//         let r = UInt8(clamping: Int(data[i*3 + 0]))
//         let g = UInt8(clamping: Int(data[i*3 + 1]))
//         let b = UInt8(clamping: Int(data[i*3 + 2]))
        
//         uint8Data[i*4 + 0] = r
//         uint8Data[i*4 + 1] = g
//         uint8Data[i*4 + 2] = b
//         uint8Data[i*4 + 3] = 255
//     }
    
//     let colorSpace = CGColorSpaceCreateDeviceRGB()
    
//     guard let context = CGContext(
//         data: &uint8Data,
//         width: width,
//         height: height,
//         bitsPerComponent: 8,
//         bytesPerRow: width * 4,
//         space: colorSpace,
//         bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
//     ),
//     let cgImage = context.makeImage()
//     else {
//         return nil
//     }
    
//     // resize
//     let targetW = targetSize
//     let targetH = targetSize
    
//     var resizedData = [UInt8](repeating: 0, count: targetW * targetH * 4)
    
//     guard let resizeContext = CGContext(
//         data: &resizedData,
//         width: targetW,
//         height: targetH,
//         bitsPerComponent: 8,
//         bytesPerRow: targetW * 4,
//         space: colorSpace,
//         bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
//     ) else {
//         return nil
//     }
    
//     resizeContext.interpolationQuality = .high
//     resizeContext.draw(cgImage, in: CGRect(x: 0, y: 0, width: targetW, height: targetH))
    
//     // UInt8 → Float32
//     var floatData: [Float32] = []
//     floatData.reserveCapacity(targetW * targetH * 3)
    
//     for i in stride(from: 0, to: resizedData.count, by: 4) {
//         floatData.append(Float32(resizedData[i]))
//         floatData.append(Float32(resizedData[i + 1]))
//         floatData.append(Float32(resizedData[i + 2]))
//     }
    
//     return ImageTensor(data: floatData, width: targetW, height: targetH)
// }
