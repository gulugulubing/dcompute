module dcompute.driver.metal.program;

import dcompute.driver.metal.device;
import std.string : toStringz;
import metal;
import foundation;

struct Library {
    private void* raw_;

    @property MTLLibrary raw() {
        return cast(MTLLibrary) raw_;
    }

    @property void raw(MTLLibrary lib) {
        raw_ = cast(void*) lib;
    }

    void release() {
        auto lib = raw();
        if (lib !is null) lib.release();
        raw_ = null;
    }

    MTLFunction newFunction(string name) {
        auto lib = raw();
        if (lib is null) return null;
        auto nsStr = NSString.create(toStringz(name));
        scope(exit) nsStr.release();
        return lib.newFunctionWithName(nsStr);
    }
}

struct Pipeline {
    private void* raw_;

    @property MTLComputePipelineState raw() {
        return cast(MTLComputePipelineState) raw_;
    }

    @property void raw(MTLComputePipelineState pso) {
        raw_ = cast(void*) pso;
    }

    void release() {
        auto pso = raw();
        if (pso !is null) pso.release();
        raw_ = null;
    }
}

struct Kernel {
    Pipeline pipeline;

    void release() {
        pipeline.release();
    }
}

struct Program {
    Device device;
    Library library;

    static Program fromDefaultDevice() {
        Program p;
        p.device = defaultDevice();
        return p;
    }

    Library loadLibrary(string path) {
        auto dev = device.raw;
        if (dev is null) return library;
        auto nsPath = NSString.create(toStringz(path));
        scope(exit) nsPath.release();
        auto url = NSURL.fromPath(nsPath);
        NSError error;
        library.raw = dev.newLibrary(url, error);
        return library;
    }

    Pipeline makePipeline(MTLFunction fn) {
        Pipeline pso;
        auto dev = device.raw;
        if (dev is null || fn is null) return pso;
        NSError error;
        pso.raw = dev.newComputePipelineStateWithFunction(
            fn, MTLPipelineOption.None, null, error);
        return pso;
    }

    Kernel getKernel(string name) {
        Kernel k;
        auto lib = library.raw;
        if (lib is null) return k;
        auto nsStr = NSString.create(toStringz(name));
        scope(exit) nsStr.release();
        auto fn = lib.newFunctionWithName(nsStr);
        if (fn is null) return k;
        k.pipeline = makePipeline(fn);
        fn.release();
        return k;
    }

    void release() {
        library.release();
    }
}
