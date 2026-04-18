import Foundation
import UIKit
import CoreGraphics

func preprocessImage(_ image: UIImage) -> [Float32]? {
    
    let targetSize = 336
    
    guard let cgImage = image.cgImage else { return nil }
    
    let srcW = cgImage.width
    let srcH = cgImage.height
    
    // CLIP normalization
    let mean: [Float32] = [0.48145466, 0.4578275, 0.40821073]
    let std:  [Float32] = [0.26862954, 0.26130258, 0.27577711]
    
    let bytesPerPixel = 4
    let bytesPerRow = targetSize * bytesPerPixel
    
    // -------------------------------------------------
    // 1. single-pass resize + padding
    // -------------------------------------------------
    var resizedData = [UInt8](repeating: 0, count: targetSize * targetSize * 4)
    
    guard let ctx = resizedData.withUnsafeMutableBytes({ ptr in
        CGContext(
            data: ptr.baseAddress,
            width: targetSize,
            height: targetSize,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        )
    }) else {
        return nil
    }
    
    // background (mean color)
    ctx.setFillColor(
        red: CGFloat(mean[0]),
        green: CGFloat(mean[1]),
        blue: CGFloat(mean[2]),
        alpha: 1.0
    )
    ctx.fill(CGRect(x: 0, y: 0, width: targetSize, height: targetSize))
    
    // aspect ratio 유지 scaling
    let scale = min(
        CGFloat(targetSize) / CGFloat(srcW),
        CGFloat(targetSize) / CGFloat(srcH)
    )
    
    let drawW = CGFloat(srcW) * scale
    let drawH = CGFloat(srcH) * scale
    
    let x = (CGFloat(targetSize) - drawW) / 2.0
    let y = (CGFloat(targetSize) - drawH) / 2.0
    
    // high quality interpolation ~= Bicubic
    ctx.interpolationQuality = .high
    
    ctx.draw(
        cgImage,
        in: CGRect(x: x, y: y, width: drawW, height: drawH)
    )
    
    // -------------------------------------------------
    // 2. normalize + CHW
    // -------------------------------------------------
    let pixelCount = targetSize * targetSize
    let inv255: Float32 = 1.0 / 255.0
    
    var output = [Float32](repeating: 0, count: 3 * pixelCount)
    
    for i in 0..<pixelCount {
        let base = i * 4
        
        let r = Float32(resizedData[base]) * inv255
        let g = Float32(resizedData[base + 1]) * inv255
        let b = Float32(resizedData[base + 2]) * inv255
        
        output[i] = (r - mean[0]) / std[0]
        output[i + pixelCount] = (g - mean[1]) / std[1]
        output[i + pixelCount * 2] = (b - mean[2]) / std[2]
    }
    
    return output
}
