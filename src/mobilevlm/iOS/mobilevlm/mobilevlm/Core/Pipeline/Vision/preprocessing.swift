import UIKit
import CoreML
import CoreGraphics

func preprocessImage(
    _ uiImage: UIImage
) -> MLMultiArray {

    guard let cgImage = uiImage.cgImage else {
        fatalError("❌ cgImage conversion failed")
    }

    let width = cgImage.width
    let height = cgImage.height

    let squareSize = max(width, height)

    // =========================================
    // Create CGContext
    // =========================================

    guard let context = CGContext(
        data: nil,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: width * 4,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else {
        fatalError("❌ CGContext creation failed")
    }

    context.draw(
        cgImage,
        in: CGRect(
            x: 0,
            y: 0,
            width: width,
            height: height
        )
    )

    guard let dataPtr = context.data else {
        fatalError("❌ CGContext data access failed")
    }

    let rgba = dataPtr.bindMemory(
        to: UInt8.self,
        capacity: width * height * 4
    )

    // =========================================
    // MLMultiArray
    // Shape:
    // [1, 3, H, W]
    // =========================================
    
    let targetSize = 336

    let mlArray = try! MLMultiArray(
        shape: [
            1,
            3,
            NSNumber(value: targetSize),
            NSNumber(value: targetSize)
        ],
        dataType: .float32
    )

    let ptr = mlArray.dataPointer.bindMemory(
        to: Float32.self,
        capacity: targetSize * targetSize * 3
    )

    let pixelCount = targetSize * targetSize

    // =========================================
    // CLIP normalization
    // =========================================

    let meanR: Float32 = 0.48145466
    let meanG: Float32 = 0.4578275
    let meanB: Float32 = 0.40821073

    let stdR: Float32 = 0.26862954
    let stdG: Float32 = 0.26130258
    let stdB: Float32 = 0.27577711

    let inv255: Float32 = 1.0 / 255.0

    let rScale = inv255 / stdR
    let gScale = inv255 / stdG
    let bScale = inv255 / stdB

    let rBias = -meanR / stdR
    let gBias = -meanG / stdG
    let bBias = -meanB / stdB

    // =========================================
    // Background color
    // =========================================

    let bgR = Float32(UInt8(meanR * 255.0))
    let bgG = Float32(UInt8(meanG * 255.0))
    let bgB = Float32(UInt8(meanB * 255.0))

    // =========================================
    // Padding offsets
    // =========================================

    let xOffset =
        (width < height)
        ? (height - width) / 2
        : 0

    let yOffset =
        (height < width)
        ? (width - height) / 2
        : 0

    // =========================================
    // Resize scale
    // =========================================

    let scale =
        Float32(squareSize) / Float32(targetSize)

    // =========================================
    // UInt8 -> Float32 LUT
    // =========================================

    var lut = [Float32](
        repeating: 0,
        count: 256
    )

    for i in 0..<256 {
        lut[i] = Float32(i)
    }

    // =========================================
    // Resize + Normalize + CHW
    // =========================================

    for y in 0..<targetSize {

        let fy =
            (Float32(y) + 0.5) * scale - 0.5

        var y0 = Int(fy)
        y0 = max(y0, 0)

        let y1 =
            min(y0 + 1, squareSize - 1)

        let dy = fy - Float32(y0)
        let idy = 1.0 - dy

        for x in 0..<targetSize {

            let fx =
                (Float32(x) + 0.5) * scale - 0.5

            var x0 = Int(fx)
            x0 = max(x0, 0)

            let x1 =
                min(x0 + 1, squareSize - 1)

            let dx = fx - Float32(x0)
            let idxf = 1.0 - dx

            // =========================================
            // Coordinates
            // =========================================

            let ox0 = x0 - xOffset
            let oy0 = y0 - yOffset

            let ox1 = x1 - xOffset
            let oy1 = y1 - yOffset

            // =========================================
            // TL
            // =========================================

            var tlR = bgR
            var tlG = bgG
            var tlB = bgB

            if ox0 >= 0 && ox0 < width &&
                oy0 >= 0 && oy0 < height {

                let idx =
                    (oy0 * width + ox0) * 4

                tlR = lut[Int(rgba[idx])]
                tlG = lut[Int(rgba[idx + 1])]
                tlB = lut[Int(rgba[idx + 2])]
            }

            // =========================================
            // TR
            // =========================================

            var trR = bgR
            var trG = bgG
            var trB = bgB

            if ox1 >= 0 && ox1 < width &&
                oy0 >= 0 && oy0 < height {

                let idx =
                    (oy0 * width + ox1) * 4

                trR = lut[Int(rgba[idx])]
                trG = lut[Int(rgba[idx + 1])]
                trB = lut[Int(rgba[idx + 2])]
            }

            // =========================================
            // BL
            // =========================================

            var blR = bgR
            var blG = bgG
            var blB = bgB

            if ox0 >= 0 && ox0 < width &&
                oy1 >= 0 && oy1 < height {

                let idx =
                    (oy1 * width + ox0) * 4

                blR = lut[Int(rgba[idx])]
                blG = lut[Int(rgba[idx + 1])]
                blB = lut[Int(rgba[idx + 2])]
            }

            // =========================================
            // BR
            // =========================================

            var brR = bgR
            var brG = bgG
            var brB = bgB

            if ox1 >= 0 && ox1 < width &&
                oy1 >= 0 && oy1 < height {

                let idx =
                    (oy1 * width + ox1) * 4

                brR = lut[Int(rgba[idx])]
                brG = lut[Int(rgba[idx + 1])]
                brB = lut[Int(rgba[idx + 2])]
            }

            // =========================================
            // Bilinear interpolation
            // =========================================

            let r =
                (
                    (tlR * idxf + trR * dx) * idy
                    +
                    (blR * idxf + brR * dx) * dy
                )

            let g =
                (
                    (tlG * idxf + trG * dx) * idy
                    +
                    (blG * idxf + brG * dx) * dy
                )

            let b =
                (
                    (tlB * idxf + trB * dx) * idy
                    +
                    (blB * idxf + brB * dx) * dy
                )

            // =========================================
            // Normalize
            // =========================================

            let nr = r * rScale + rBias
            let ng = g * gScale + gBias
            let nb = b * bScale + bBias

            let outIdx =
                y * targetSize + x

            ptr[outIdx] = nr
            ptr[outIdx + pixelCount] = ng
            ptr[outIdx + pixelCount * 2] = nb
        }
    }

    return mlArray
}
