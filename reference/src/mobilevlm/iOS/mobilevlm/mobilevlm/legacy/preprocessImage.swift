//import Foundation
//import UIKit
//import ImageIO
//
//
//
//func extractRGB(from uiImage: UIImage) -> [UInt8]? {
//
//    guard let cgImage = uiImage.cgImage else {
//        print("❌ cgImage conversion failed")
//        return nil
//    }
//
//    let width = cgImage.width
//    let height = cgImage.height
//
//    let bytesPerPixel = 4
//    let bytesPerRow = width * bytesPerPixel
//    let totalBytes = height * bytesPerRow
//
//    var rgba = [UInt8](repeating: 0, count: totalBytes)
//
//    let colorSpace = CGColorSpaceCreateDeviceRGB()
//    let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
//
//    guard let context = CGContext(
//        data: &rgba,
//        width: width,
//        height: height,
//        bitsPerComponent: 8,
//        bytesPerRow: bytesPerRow,
//        space: colorSpace,
//        bitmapInfo: bitmapInfo
//    ) else {
//        print("❌ CGContext failed")
//        return nil
//    }
//
//    context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
//
//    var rgb = [UInt8]()
//    rgb.reserveCapacity(width * height * 3)
//
//    for i in stride(from: 0, to: rgba.count, by: 4) {
//        rgb.append(rgba[i])     // R
//        rgb.append(rgba[i + 1]) // G
//        rgb.append(rgba[i + 2]) // B
//    }
//
//    return rgb
//}
//
//
//
//func expandToSquare(
//    rgb: [UInt8],
//    width: Int,
//    height: Int,
//) -> [UInt8] {
//
//    if width == height {
//        return (rgb)
//    }
//    
//    let clipMean: [Float] = [0.48145466, 0.4578275, 0.40821073]
//    let bgColor: [UInt8] = clipMean.map {
//        UInt8(floor($0 * 255.0))
//    }
//
//    let size = max(width, height)
//    let channels = 3
//
//    // result buffer
//    var result = [UInt8](
//        repeating: 0,
//        count: size * size * channels
//    )
//
//    // fill with bg color
//    for i in stride(from: 0, to: result.count, by: 3) {
//        result[i]     = bgColor[0]
//        result[i + 1] = bgColor[1]
//        result[i + 2] = bgColor[2]
//    }
//
//    // offset 계산
//    let xOffset = (width < height) ? (height - width) / 2 : 0
//    let yOffset = (height < width) ? (width - height) / 2 : 0
//
//    // copy original image → result
//    for y in 0..<height {
//        for x in 0..<width {
//
//            let srcIdx = (y * width + x) * 3
//            let dstIdx = ((y + yOffset) * size + (x + xOffset)) * 3
//
//            result[dstIdx]     = rgb[srcIdx]
//            result[dstIdx + 1] = rgb[srcIdx + 1]
//            result[dstIdx + 2] = rgb[srcIdx + 2]
//        }
//    }
//
//    return result
//}
//
//
//
//func resizeBilinear336(
//    img: [UInt8],
//    size: Int  // input is square (from expandToSquare)
//) -> [UInt8] {
//
//    let outSize = 336
//    let channels = 3
//
//    let inH = size
//    let inW = size
//
//    var output = [Float](
//        repeating: 0.0,
//        count: outSize * outSize * channels
//    )
//
//    let scaleX = Float(inW) / Float(outSize)
//    let scaleY = Float(inH) / Float(outSize)
//
//    for y in 0..<outSize {
//        for x in 0..<outSize {
//
//            // pixel center align
//            let srcX = (Float(x) + 0.5) * scaleX - 0.5
//            let srcY = (Float(y) + 0.5) * scaleY - 0.5
//
//            var x0 = Int(floor(srcX))
//            var y0 = Int(floor(srcY))
//
//            let x1 = min(x0 + 1, inW - 1)
//            let y1 = min(y0 + 1, inH - 1)
//
//            let dx = srcX - Float(x0)
//            let dy = srcY - Float(y0)
//
//            // boundary clamp (same order as Python)
//            x0 = max(x0, 0)
//            y0 = max(y0, 0)
//
//            let idxTL = (y0 * inW + x0) * 3
//            let idxTR = (y0 * inW + x1) * 3
//            let idxBL = (y1 * inW + x0) * 3
//            let idxBR = (y1 * inW + x1) * 3
//
//            for c in 0..<3 {
//
//                let topLeft     = Float(img[idxTL + c])
//                let topRight    = Float(img[idxTR + c])
//                let bottomLeft  = Float(img[idxBL + c])
//                let bottomRight = Float(img[idxBR + c])
//
//                let top = topLeft * (1.0 - dx) + topRight * dx
//                let bottom = bottomLeft * (1.0 - dx) + bottomRight * dx
//                let value = top * (1.0 - dy) + bottom * dy
//
//                let outIdx = (y * outSize + x) * 3 + c
//                output[outIdx] = value
//            }
//        }
//    }
//
//    // Float → UInt8 (clip)
//    var result = [UInt8](repeating: 0, count: output.count)
//
//    for i in 0..<output.count {
//        let v = max(0.0, min(255.0, output[i]))
//        result[i] = UInt8(v)
//    }
//
//    return result
//}
//
//
//func normalizeAndConvertToCHW(
//    rgb: [UInt8],
//    width: Int,
//    height: Int
//) -> [Float] {
//
//    let pixelCount = width * height
//
//    let clipMean: [Float] = [0.48145466, 0.4578275, 0.40821073]
//    let clipStd:  [Float] = [0.26862954, 0.26130258, 0.27577711]
//
//    // CHW buffer
//    var chw = [Float](repeating: 0, count: pixelCount * 3)
//
//    for i in 0..<pixelCount {
//
//        let r = Float(rgb[i * 3]) / 255.0
//        let g = Float(rgb[i * 3 + 1]) / 255.0
//        let b = Float(rgb[i * 3 + 2]) / 255.0
//
//        // normalize
//        let nr = (r - clipMean[0]) / clipStd[0]
//        let ng = (g - clipMean[1]) / clipStd[1]
//        let nb = (b - clipMean[2]) / clipStd[2]
//
//        // CHW layout
//        chw[i] = nr                          // R channel
//        chw[i + pixelCount] = ng            // G channel
//        chw[i + pixelCount * 2] = nb        // B channel
//    }
//
//    return chw
//}
