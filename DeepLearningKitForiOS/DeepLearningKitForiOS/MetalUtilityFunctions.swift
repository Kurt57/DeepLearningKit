//
//  MetalUtilFunctions.swift
//  MemkiteMetal
//
//  Created by Amund Tveit & Torb Morland on 24/11/15.
//  Copyright © 2015 Memkite. All rights reserved.
//

import Foundation
import Metal

func createComplexNumbersArray(count: Int) -> [MetalComplexNumberType] {
    let zeroComplexNumber = MetalComplexNumberType()
    return [MetalComplexNumberType](repeating: zeroComplexNumber, count: count)
}

func createFloatNumbersArray(count: Int) -> [Float] {
    return [Float](repeating: 0.0, count: count)
}

func createFloatMetalBuffer( vector: [Float], metalDevice:MTLDevice) -> MTLBuffer {
    var vector = vector

    let byteLength = vector.count*MemoryLayout<Float>.size // future: MTLResourceStorageModePrivate
    return metalDevice.makeBuffer(bytes: &vector, length: byteLength, options: [])
}

func createComplexMetalBuffer( vector:[MetalComplexNumberType], metalDevice:MTLDevice) -> MTLBuffer {
    var vector = vector
    let byteLength = vector.count*MemoryLayout<MetalComplexNumberType>.size // or size of and actual 1st element object?
    return metalDevice.makeBuffer(bytes: &vector, length: byteLength, options: [])
}

func createShaderParametersMetalBuffer( shaderParameters:MetalShaderParameters,  metalDevice:MTLDevice) -> MTLBuffer {
    var shaderParameters = shaderParameters
    let byteLength = MemoryLayout<MetalShaderParameters>.size
    return metalDevice.makeBuffer(bytes: &shaderParameters, length: byteLength, options: [])
}

func createMatrixShaderParametersMetalBuffer( params: MetalMatrixVectorParameters,  metalDevice: MTLDevice) -> MTLBuffer {
    var params = params
    let byteLength = MemoryLayout<MetalMatrixVectorParameters>.size
    return metalDevice.makeBuffer(bytes: &params, length: byteLength, options: [])
    
}

func createPoolingParametersMetalBuffer( params: MetalPoolingParameters, metalDevice: MTLDevice) -> MTLBuffer {
    var params = params
    let byteLength = MemoryLayout<MetalPoolingParameters>.size
    return metalDevice.makeBuffer(bytes: &params, length: byteLength, options: [])
}

func createConvolutionParametersMetalBuffer( params: MetalConvolutionParameters, metalDevice: MTLDevice) -> MTLBuffer {
    var params = params
    let byteLength = MemoryLayout<MetalConvolutionParameters>.size
    return metalDevice.makeBuffer(bytes: &params, length: byteLength, options: [])
}

func createTensorDimensionsVectorMetalBuffer( vector: [MetalTensorDimensions], metalDevice: MTLDevice) -> MTLBuffer {
    var vector = vector
    let byteLength = vector.count * MemoryLayout<MetalTensorDimensions>.size
    return metalDevice.makeBuffer(bytes: &vector, length: byteLength, options: [])
}

func setupShaderInMetalPipeline(shaderName:String, metalDefaultLibrary:MTLLibrary, metalDevice:MTLDevice) -> (shader:MTLFunction?,
    computePipelineState:MTLComputePipelineState?,
    computePipelineErrors:NSErrorPointer?)  {
        let shader = metalDefaultLibrary.makeFunction(name: shaderName)
        let computePipeLineDescriptor = MTLComputePipelineDescriptor()
        computePipeLineDescriptor.computeFunction = shader
        //        var computePipelineErrors = NSErrorPointer()
        //            let computePipelineState:MTLComputePipelineState = metalDevice.newComputePipelineStateWithFunction(shader!, completionHandler: {(})
        let computePipelineErrors = NSErrorPointer.init(nilLiteral: ())
        var computePipelineState:MTLComputePipelineState? = nil
        do {
            computePipelineState = try metalDevice.makeComputePipelineState(function: shader!)
        } catch {
            print("catching..")
        }
        return (shader, computePipelineState, computePipelineErrors)
        
}

func createMetalBuffer( vector:[Float], metalDevice:MTLDevice) -> MTLBuffer {
    var vector = vector
    let byteLength = vector.count*MemoryLayout<Float>.size
    return metalDevice.makeBuffer(bytes: &vector, length: byteLength, options: [])
}

func preLoadMetalShaders(metalDevice: MTLDevice, metalDefaultLibrary: MTLLibrary) {
    let shaders = ["avg_pool", "max_pool", "rectifier_linear", "convolution_layer", "im2col"]
    for shader in shaders {
        setupShaderInMetalPipeline(shaderName: shader, metalDefaultLibrary: metalDefaultLibrary,metalDevice: metalDevice) // TODO: this returns stuff
    }
}

func createOrReuseFloatMetalBuffer(name:String, data: [Float], cache:inout [Dictionary<String,MTLBuffer>], layer_number:Int, metalDevice:MTLDevice) -> MTLBuffer {
    var result:MTLBuffer
    if let tmpval = cache[layer_number][name] {
        print("found key = \(name) in cache")
        result = tmpval
    } else {
        print("didnt find key = \(name) in cache")
        result = createFloatMetalBuffer(vector: data, metalDevice: metalDevice)
        cache[layer_number][name] = result
        // print("DEBUG: cache = \(cache)")
    }
    
    return result
}


func createOrReuseConvolutionParametersMetalBuffer(name:String,
    data: MetalConvolutionParameters,
    cache:inout [Dictionary<String,MTLBuffer>], layer_number: Int, metalDevice: MTLDevice) -> MTLBuffer {
        var result:MTLBuffer
        if let tmpval = cache[layer_number][name] {
           print("found key = \(name) in cache")
            result = tmpval
        } else {
            print("didnt find key = \(name) in cache")
            result = createConvolutionParametersMetalBuffer(params: data, metalDevice: metalDevice)
            cache[layer_number][name] = result
            //print("DEBUG: cache = \(cache)")
        }
        
        return result
}

func createOrReuseTensorDimensionsVectorMetalBuffer(name:String,
    data:[MetalTensorDimensions], cache:inout [Dictionary<String,MTLBuffer>], layer_number: Int, metalDevice: MTLDevice) -> MTLBuffer {
        var result:MTLBuffer
        if let tmpval = cache[layer_number][name] {
            print("found key = \(name) in cache")
            result = tmpval
        } else {
            print("didnt find key = \(name) in cache")
            result = createTensorDimensionsVectorMetalBuffer(vector: data, metalDevice: metalDevice)
            cache[layer_number][name] = result
            //print("DEBUG: cache = \(cache)")
        }
        
        return result
}

//
//let sizeParamMetalBuffer = createShaderParametersMetalBuffer(size_params, metalDevice: metalDevice)
//let poolingParamMetalBuffer = createPoolingParametersMetalBuffer(pooling_params, metalDevice: metalDevice)

func createOrReuseShaderParametersMetalBuffer(name:String,
    data:MetalShaderParameters, cache:inout [Dictionary<String,MTLBuffer>], layer_number: Int, metalDevice: MTLDevice) -> MTLBuffer {
        var result:MTLBuffer
        if let tmpval = cache[layer_number][name] {
//            print("found key = \(name) in cache")
            result = tmpval
        } else {
//            print("didnt find key = \(name) in cache")
            result = createShaderParametersMetalBuffer(shaderParameters: data, metalDevice: metalDevice)
            cache[layer_number][name] = result
            //print("DEBUG: cache = \(cache)")
        }
        
        return result
}

func createOrReusePoolingParametersMetalBuffer(name:String,
    data:MetalPoolingParameters, cache:inout [Dictionary<String,MTLBuffer>], layer_number: Int, metalDevice: MTLDevice) -> MTLBuffer {
        var result:MTLBuffer
        if let tmpval = cache[layer_number][name] {
//            print("found key = \(name) in cache")
            result = tmpval
        } else {
//            print("didnt find key = \(name) in cache")
            result = createPoolingParametersMetalBuffer(params: data, metalDevice: metalDevice)
            cache[layer_number][name] = result
            //print("DEBUG: cache = \(cache)")
        }
        
        return result
}


