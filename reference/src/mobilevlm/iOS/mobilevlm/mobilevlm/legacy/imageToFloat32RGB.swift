//
//  imagetofloat32.swift
//  mobilevlm
//
//  Created by hyeongseob jo on 4/18/26.
//

import Foundation
import UIKit

struct ImageTensor {
    let data: [Float32]
    let width: Int
    let height: Int
}

func imageToFloat32RGB(_ image: UIImage) -> ImageTensor? {
    guard let cgImage = image.cgImage else { return nil }
    
    let width = cgImage.width
    let height = cgImage.height
    let bytesPerPixel = 4
    let bytesPerRow = bytesPerPixel * width
    
    var pixelData = [UInt8](repeating: 0, count: Int(height * bytesPerRow))
    
    guard let context = CGContext(
        data: &pixelData,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: bytesPerRow,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
    ) else {
        return nil
    }
    
    context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
    
    // ✅ RGBA → RGB 변환
    var rgbArray: [Float32] = []
    rgbArray.reserveCapacity(width * height * 3)
    
    for i in stride(from: 0, to: pixelData.count, by: 4) {
        rgbArray.append(Float32(pixelData[i]))     // R
        rgbArray.append(Float32(pixelData[i + 1])) // G
        rgbArray.append(Float32(pixelData[i + 2])) // B
    }
    
    return ImageTensor(data: rgbArray, width: width, height: height)
}
