import UIKit
import CoreML
import CoreGraphics

func preprocessImage(
    _ uiImage: UIImage,
    targetSize: Int = 336
) -> MLMultiArray? {

    guard let cgImage = uiImage.cgImage else {
        print("❌ cgImage conversion failed")
        return nil
    }

    let width = cgImage.width
    let height = cgImage.height

    let squareSize = max(width, height)

    // =========================================
    // RGBA extraction
    // =========================================

    let bytesPerPixel = 4
    let bytesPerRow = width * bytesPerPixel
    let totalBytes = height * bytesPerRow

    var rgba = [UInt8](repeating: 0, count: totalBytes)

    guard let context = CGContext(
        data: &rgba,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: bytesPerRow,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else {

        print("❌ CGContext creation failed")
        return nil
    }

    context.draw(
        cgImage,
        in: CGRect(x: 0, y: 0, width: width, height: height)
    )

    // =========================================
    // MLMultiArray
    // =========================================

    let pixelCount = targetSize * targetSize

    guard let mlArray = try? MLMultiArray(
        shape: [1, 3, targetSize as NSNumber, targetSize as NSNumber],
        dataType: .float32
    ) else {

        print("❌ MLMultiArray creation failed")
        return nil
    }

    let ptr = mlArray.dataPointer.bindMemory(
        to: Float32.self,
        capacity: pixelCount * 3
    )

    // =========================================
    // CLIP normalization constants
    // =========================================

    let meanR: Float32 = 0.48145466
    let meanG: Float32 = 0.4578275
    let meanB: Float32 = 0.40821073

    let stdR: Float32 = 0.26862954
    let stdG: Float32 = 0.26130258
    let stdB: Float32 = 0.27577711

    // background color
    let bgR = Float32(UInt8(floor(meanR * 255.0)))
    let bgG = Float32(UInt8(floor(meanG * 255.0)))
    let bgB = Float32(UInt8(floor(meanB * 255.0)))

    // padding offsets
    let xOffset = (width < height)
        ? (height - width) / 2
        : 0

    let yOffset = (height < width)
        ? (width - height) / 2
        : 0

    // resize scale
    let scaleX = Float32(squareSize) / Float32(targetSize)
    let scaleY = Float32(squareSize) / Float32(targetSize)

    // =========================================
    // Resize + Normalize + CHW
    // =========================================

    for y in 0..<targetSize {

        for x in 0..<targetSize {

            // =================================
            // bilinear source coordinate
            // =================================

            let srcX = (Float32(x) + 0.5) * scaleX - 0.5
            let srcY = (Float32(y) + 0.5) * scaleY - 0.5

            var x0 = Int(floor(Double(srcX)))
            var y0 = Int(floor(Double(srcY)))

            let x1 = min(x0 + 1, squareSize - 1)
            let y1 = min(y0 + 1, squareSize - 1)

            let dx = srcX - Float32(x0)
            let dy = srcY - Float32(y0)

            x0 = max(x0, 0)
            y0 = max(y0, 0)

            // =================================
            // helper
            // =================================

            func sample(_ sx: Int, _ sy: Int) -> (Float32, Float32, Float32) {

                let ox = sx - xOffset
                let oy = sy - yOffset

                // padded region
                if ox < 0 || ox >= width || oy < 0 || oy >= height {

                    return (bgR, bgG, bgB)
                }

                let rgbaIndex = (oy * width + ox) * 4

                return (
                    Float32(rgba[rgbaIndex]),
                    Float32(rgba[rgbaIndex + 1]),
                    Float32(rgba[rgbaIndex + 2])
                )
            }

            // =================================
            // 4 points
            // =================================

            let tl = sample(x0, y0)
            let tr = sample(x1, y0)
            let bl = sample(x0, y1)
            let br = sample(x1, y1)

            // =================================
            // bilinear interpolation
            // =================================

            let rTop = tl.0 * (1 - dx) + tr.0 * dx
            let rBottom = bl.0 * (1 - dx) + br.0 * dx
            let r = rTop * (1 - dy) + rBottom * dy

            let gTop = tl.1 * (1 - dx) + tr.1 * dx
            let gBottom = bl.1 * (1 - dx) + br.1 * dx
            let g = gTop * (1 - dy) + gBottom * dy

            let bTop = tl.2 * (1 - dx) + tr.2 * dx
            let bBottom = bl.2 * (1 - dx) + br.2 * dx
            let b = bTop * (1 - dy) + bBottom * dy

            // =================================
            // normalize
            // =================================

            let nr = (r / 255.0 - meanR) / stdR
            let ng = (g / 255.0 - meanG) / stdG
            let nb = (b / 255.0 - meanB) / stdB

            // =================================
            // CHW
            // =================================

            let idx = y * targetSize + x

            ptr[idx] = nr
            ptr[idx + pixelCount] = ng
            ptr[idx + pixelCount * 2] = nb
        }
    }

    return mlArray
}
