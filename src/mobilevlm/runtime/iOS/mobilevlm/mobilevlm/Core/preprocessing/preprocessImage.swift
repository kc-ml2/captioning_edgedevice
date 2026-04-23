import Foundation
import UIKit


func extractRGB(from uiImage: UIImage) -> [UInt8]? {

    guard let cgImage = uiImage.cgImage else {
        print("❌ cgImage conversion failed")
        return nil
    }

    let width = cgImage.width
    let height = cgImage.height

    let bytesPerPixel = 4
    let bytesPerRow = width * bytesPerPixel
    let totalBytes = height * bytesPerRow

    var rgba = [UInt8](repeating: 0, count: totalBytes)

    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue

    guard let context = CGContext(
        data: &rgba,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: bytesPerRow,
        space: colorSpace,
        bitmapInfo: bitmapInfo
    ) else {
        print("❌ CGContext failed")
        return nil
    }

    context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

    var rgb = [UInt8]()
    rgb.reserveCapacity(width * height * 3)

    for i in stride(from: 0, to: rgba.count, by: 4) {
        rgb.append(rgba[i])     // R
        rgb.append(rgba[i + 1]) // G
        rgb.append(rgba[i + 2]) // B
    }

    return rgb
}


func expandToSquare(
    rgb: [UInt8],
    width: Int,
    height: Int,
) -> [UInt8] {

    if width == height {
        return (rgb)
    }
    
    let clipMean: [Float] = [0.48145466, 0.4578275, 0.40821073]
    let bgColor: [UInt8] = clipMean.map {
        UInt8(floor($0 * 255.0))
    }

    let size = max(width, height)
    let channels = 3

    // result buffer
    var result = [UInt8](
        repeating: 0,
        count: size * size * channels
    )

    // fill with bg color
    for i in stride(from: 0, to: result.count, by: 3) {
        result[i]     = bgColor[0]
        result[i + 1] = bgColor[1]
        result[i + 2] = bgColor[2]
    }

    // offset 계산
    let xOffset = (width < height) ? (height - width) / 2 : 0
    let yOffset = (height < width) ? (width - height) / 2 : 0

    // copy original image → result
    for y in 0..<height {
        for x in 0..<width {

            let srcIdx = (y * width + x) * 3
            let dstIdx = ((y + yOffset) * size + (x + xOffset)) * 3

            result[dstIdx]     = rgb[srcIdx]
            result[dstIdx + 1] = rgb[srcIdx + 1]
            result[dstIdx + 2] = rgb[srcIdx + 2]
        }
    }

    return result
}

func rgbToUIImage(rgb: [UInt8], width: Int, height: Int) -> UIImage? {

    let bytesPerPixel = 4
    let bytesPerRow = width * bytesPerPixel

    var rgba = [UInt8](repeating: 255, count: width * height * 4)

    for i in 0..<(width * height) {
        rgba[i * 4]     = rgb[i * 3]
        rgba[i * 4 + 1] = rgb[i * 3 + 1]
        rgba[i * 4 + 2] = rgb[i * 3 + 2]
        rgba[i * 4 + 3] = 255
    }

    let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!

    guard let context = CGContext(
        data: &rgba,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: bytesPerRow,
        space: colorSpace,
        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
    ),
    let cgImage = context.makeImage() else {
        return nil
    }

    return UIImage(cgImage: cgImage)
}

import CoreImage

func resizeWithCI(_ image: UIImage) -> UIImage? {

    guard let ciImage = CIImage(image: image) else { return nil }

    // 🔥 Python과 동일한 boundary 처리
    let clamped = ciImage.clampedToExtent()

    let scale = 336.0 / ciImage.extent.width

    guard let filter = CIFilter(name: "CILanczosScaleTransform") else {
        return nil
    }

    filter.setValue(clamped, forKey: kCIInputImageKey)
    filter.setValue(scale, forKey: kCIInputScaleKey)
    filter.setValue(1.0, forKey: kCIInputAspectRatioKey)

    let context = CIContext(options: [
        .workingColorSpace: CGColorSpace(name: CGColorSpace.sRGB)!,
        .outputColorSpace: CGColorSpace(name: CGColorSpace.sRGB)!
    ])

    guard let output = filter.outputImage,
          let cgImage = context.createCGImage(
            output,
            from: CGRect(x: 0, y: 0, width: 336, height: 336)
          ) else {
        return nil
    }

    return UIImage(cgImage: cgImage)
}


func normalizeAndConvertToCHW(
    rgb: [UInt8],
    width: Int,
    height: Int
) -> [Float] {

    let pixelCount = width * height

    let clipMean: [Float] = [0.48145466, 0.4578275, 0.40821073]
    let clipStd:  [Float] = [0.26862954, 0.26130258, 0.27577711]

    // CHW buffer
    var chw = [Float](repeating: 0, count: pixelCount * 3)

    for i in 0..<pixelCount {

        let r = Float(rgb[i * 3]) / 255.0
        let g = Float(rgb[i * 3 + 1]) / 255.0
        let b = Float(rgb[i * 3 + 2]) / 255.0

        // normalize
        let nr = (r - clipMean[0]) / clipStd[0]
        let ng = (g - clipMean[1]) / clipStd[1]
        let nb = (b - clipMean[2]) / clipStd[2]

        // CHW layout
        chw[i] = nr                          // R channel
        chw[i + pixelCount] = ng            // G channel
        chw[i + pixelCount * 2] = nb        // B channel
    }

    return chw
}
