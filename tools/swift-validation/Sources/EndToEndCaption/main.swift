import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXNN
import SentencepieceTokenizer

let hiddenSize = 2048, heads = 16, headDimension = 128, languageLayers = 24
let imageToken = -200
struct Cache { var key: MLXArray; var value: MLXArray }
struct AppError: Error, CustomStringConvertible { let description: String }

func loadModel(_ directory: URL) throws -> [String: MLXArray] {
    let data = try Data(contentsOf: directory.appending(path: "model.safetensors.index.json"))
    let object = try JSONSerialization.jsonObject(with: data) as! [String: Any]
    let map = object["weight_map"] as! [String: String]
    var result: [String: MLXArray] = [:]
    for shard in Set(map.values).sorted() {
        print("Loading \(shard)...")
        result.merge(try loadArrays(url: directory.appending(path: shard))) { _, new in new }
    }
    return result
}

func required(_ weights: [String: MLXArray], _ name: String) throws -> MLXArray {
    guard let value = weights[name] else { throw AppError(description: "Missing weight: \(name)") }
    return value
}
func linear(_ x: MLXArray, _ w: [String: MLXArray], _ name: String) throws -> MLXArray { x.matmul(try required(w, name).transposed()) }
func affine(_ x: MLXArray, _ w: [String: MLXArray], _ prefix: String) throws -> MLXArray { try linear(x, w, "\(prefix).weight") + required(w, "\(prefix).bias") }
func rmsNorm(_ x: MLXArray, _ weight: MLXArray) -> MLXArray { x * rsqrt((x*x).mean(axis: -1, keepDims: true)+1e-6)*weight }
func layerNorm(_ x: MLXArray, _ w: [String: MLXArray], _ prefix: String) throws -> MLXArray {
    let mean=x.mean(axis:-1,keepDims:true), centered=x-mean
    return centered*rsqrt((centered*centered).mean(axis:-1,keepDims:true)+1e-5)*(try required(w,"\(prefix).weight"))+(try required(w,"\(prefix).bias"))
}

func loadPixels(_ url: URL) throws -> MLXArray {
    guard let source=CGImageSourceCreateWithURL(url as CFURL,nil), let image=CGImageSourceCreateImageAtIndex(source,0,nil) else { throw AppError(description:"Cannot read image") }
    let width=image.width, height=image.height, square=max(width,height), target=336
    let mean:[Float]=[0.48145466,0.4578275,0.40821073], std:[Float]=[0.26862954,0.26130258,0.27577711]
    let bg=[UInt8(mean[0]*255),UInt8(mean[1]*255),UInt8(mean[2]*255),UInt8(255)]
    var squarePixels=[UInt8](repeating:0,count:square*square*4)
    for i in 0..<(square*square) { squarePixels[i*4]=bg[0];squarePixels[i*4+1]=bg[1];squarePixels[i*4+2]=bg[2];squarePixels[i*4+3]=255 }
    guard let context=CGContext(data:&squarePixels,width:square,height:square,bitsPerComponent:8,bytesPerRow:square*4,space:CGColorSpaceCreateDeviceRGB(),bitmapInfo:CGImageAlphaInfo.premultipliedLast.rawValue) else { throw AppError(description:"CGContext failed") }
    context.interpolationQuality = .none
    context.draw(image,in:CGRect(x:(square-width)/2,y:(square-height)/2,width:width,height:height))
    var resized=[UInt8](repeating:0,count:target*target*4)
    guard let output=CGContext(data:&resized,width:target,height:target,bitsPerComponent:8,bytesPerRow:target*4,space:CGColorSpaceCreateDeviceRGB(),bitmapInfo:CGImageAlphaInfo.premultipliedLast.rawValue) else { throw AppError(description:"Resize context failed") }
    output.interpolationQuality = .high
    output.draw(context.makeImage()!,in:CGRect(x:0,y:0,width:target,height:target))
    var values=[Float](repeating:0,count:target*target*3)
    for y in 0..<target { for x in 0..<target { let p=(y*target+x)*4, o=(y*target+x)*3; values[o]=(Float(resized[p])/255-mean[0])/std[0]; values[o+1]=(Float(resized[p+1])/255-mean[1])/std[1]; values[o+2]=(Float(resized[p+2])/255-mean[2])/std[2] } }
    return MLXArray(values,[1,target,target,3])
}

func vision(_ pixels:MLXArray,_ w:[String:MLXArray]) throws -> MLXArray {
    let r="vision_model.vision_model"
    let patches=conv2d(pixels,try required(w,"\(r).embeddings.patch_embedding.weight"),stride:14).reshaped(1,576,1024)
    let cls=try required(w,"\(r).embeddings.class_embedding").reshaped(1,1,1024)
    var x=concatenated([cls,patches],axis:1)+(try required(w,"\(r).embeddings.position_embedding.weight"))
    x=try layerNorm(x,w,"\(r).pre_layrnorm")
    for n in 0..<23 {
        let p="\(r).encoder.layers.\(n)", y=try layerNorm(x,w,"\(p).layer_norm1")
        let q=try affine(y,w,"\(p).self_attn.q_proj").reshaped(1,577,16,64).transposed(0,2,1,3)
        let k=try affine(y,w,"\(p).self_attn.k_proj").reshaped(1,577,16,64).transposed(0,2,1,3)
        let v=try affine(y,w,"\(p).self_attn.v_proj").reshaped(1,577,16,64).transposed(0,2,1,3)
        var a=softmax(q.matmul(k.transposed(0,1,3,2))/8,axis:-1).matmul(v).transposed(0,2,1,3).reshaped(1,577,1024)
        a=try affine(a,w,"\(p).self_attn.out_proj"); x=x+a
        var m=try affine(layerNorm(x,w,"\(p).layer_norm2"),w,"\(p).mlp.fc1");m=m*sigmoid(1.702*m)
        x=x+(try affine(m,w,"\(p).mlp.fc2"));eval(x);print("Vision \(n+1)/23")
    }
    return x[0...,1...,0...]
}
func projector(_ input:MLXArray,_ w:[String:MLXArray]) throws -> MLXArray {
    var x=try affine(input,w,"projector.mlp.mlp.0");x=gelu(x);x=try affine(x,w,"projector.mlp.mlp.2")
    let pool=AvgPool2d(kernelSize:2,stride:2)(x.reshaped(1,24,24,2048))
    return (pool+conv2d(pool,try required(w,"projector.peg.peg.0.weight"),padding:1,groups:2048)+(try required(w,"projector.peg.peg.0.bias"))).reshaped(1,144,2048)
}

func rotateHalf(_ x:MLXArray)->MLXArray { concatenated([-x[0...,0...,0...,(headDimension/2)...],x[0...,0...,0...,..<(headDimension/2)]],axis:-1) }
func rope(_ x:MLXArray,_ positions:MLXArray)->MLXArray {
    let d=arange(0,headDimension,step:2).asType(.float32), inv=1.0/pow(10_000.0,d/Float(headDimension)), f=outer(positions.asType(.float32),inv)
    let e=concatenated([f,f],axis:-1).reshaped(1,1,positions.size,headDimension)
    return x*cos(e)+rotateHalf(x)*sin(e)
}
func languageLayer(_ input:MLXArray,_ w:[String:MLXArray],_ n:Int,_ positions:MLXArray,_ past:Cache?) throws ->(MLXArray,Cache) {
    let p="language_model.layers.\(n)", length=input.shape[1], y=rmsNorm(input,try required(w,"\(p).input_layernorm.weight"))
    let q=rope(try linear(y,w,"\(p).self_attn.q_proj.weight").reshaped(1,length,heads,headDimension).transposed(0,2,1,3),positions)
    let nk=rope(try linear(y,w,"\(p).self_attn.k_proj.weight").reshaped(1,length,heads,headDimension).transposed(0,2,1,3),positions)
    let nv=try linear(y,w,"\(p).self_attn.v_proj.weight").reshaped(1,length,heads,headDimension).transposed(0,2,1,3)
    let k=past.map{concatenated([$0.key,nk],axis:2)} ?? nk, v=past.map{concatenated([$0.value,nv],axis:2)} ?? nv
    var scores=q.matmul(k.transposed(0,1,3,2))/sqrt(Float(headDimension));if past==nil { scores=scores+(1.0-tri(length,dtype:.float32)) * -1e9 }
    var a=softmax(scores,axis:-1).matmul(v).transposed(0,2,1,3).reshaped(1,length,hiddenSize);a=try linear(a,w,"\(p).self_attn.o_proj.weight")
    let residual=input+a, z=rmsNorm(residual,try required(w,"\(p).post_attention_layernorm.weight"))
    let gate=silu(try linear(z,w,"\(p).mlp.gate_proj.weight")),up=try linear(z,w,"\(p).mlp.up_proj.weight")
    return (residual+(try linear(gate*up,w,"\(p).mlp.down_proj.weight")),Cache(key:k,value:v))
}

func tokenIDs(_ tokenizer:SentencepieceTokenizer,_ question:String)throws->[Int] {
    let prompt="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n\(question) ASSISTANT:"
    let chunks=prompt.components(separatedBy:"<image>");var ids:[Int]=[]
    for (index,chunk) in chunks.enumerated(){let encoded=try tokenizer.encode(chunk).map{$0-1};if index==0{ids.append(1);ids.append(contentsOf:encoded)}else{ids.append(imageToken);ids.append(contentsOf:encoded.filter{$0 != 1})}}
    return ids
}

func main() throws {
    guard CommandLine.arguments.count>=3 else {throw AppError(description:"Usage: EndToEndCaption <model-directory> <image> [question]")}
    let modelURL=URL(filePath:CommandLine.arguments[1]),imageURL=URL(filePath:CommandLine.arguments[2]),question=CommandLine.arguments.count>3 ? CommandLine.arguments[3]:"What objects are visible in the scene?"
    let weights=try loadModel(modelURL),tokenizer=try SentencepieceTokenizer(modelPath:modelURL.appending(path:"tokenizer.model").path)
    let ids=try tokenIDs(tokenizer,question),imagePosition=ids.firstIndex(of:imageToken)!,embedding=try required(weights,"language_model.embed_tokens.weight")
    print("Token count: \(ids.count)");let pixels=try loadPixels(imageURL);let imageFeatures=try projector(vision(pixels,weights),weights);eval(imageFeatures)
    let before=embedding.take(MLXArray(ids[..<imagePosition]),axis:0),after=embedding.take(MLXArray(ids[(imagePosition+1)...]),axis:0)
    var hidden=concatenated([before,imageFeatures[0],after],axis:0).expandedDimensions(axis:0),caches:[Cache]=[]
    let prefillLength=hidden.shape[1];print("Multimodal length: \(prefillLength)")
    for n in 0..<languageLayers{let r=try languageLayer(hidden,weights,n,arange(prefillLength),nil);hidden=r.0;caches.append(r.1);eval(hidden);print("Prefill \(n+1)/24")}
    var generated:[Int]=[]
    for step in 0..<40 {
        let logits=try linear(rmsNorm(hidden,required(weights,"language_model.norm.weight")),weights,"lm_head.weight")[0,hidden.shape[1]-1]
        let token=Int(logits.argMax().item(Int32.self));generated.append(token);print("Token \(step+1): \(token)");if token==2{break}
        hidden=embedding[token].reshaped(1,1,hiddenSize)
        for n in 0..<languageLayers{let r=try languageLayer(hidden,weights,n,MLXArray([caches[n].key.shape[2]]),caches[n]);hidden=r.0;caches[n]=r.1;eval(hidden)}
    }
    let caption=try tokenizer.decode(generated.map{$0+1}).replacingOccurrences(of:"\\s+",with:" ",options:.regularExpression).trimmingCharacters(in:.whitespacesAndNewlines)
    print("Generated tokens: \(generated)");print("Caption: \(caption)")
}

do{try main()}catch{FileHandle.standardError.write(Data("ERROR: \(error)\n".utf8));exit(1)}
