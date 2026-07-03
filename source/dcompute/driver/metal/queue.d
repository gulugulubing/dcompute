module dcompute.driver.metal.queue;

import dcompute.driver.metal.device;
import dcompute.driver.metal.encoder;
import dcompute.driver.metal.program;
import dcompute.driver.metal.buffer;
import foundation;
import metal;
import std.meta : allSatisfy;
import std.traits : isNumeric, Unqual;

private enum isBuffer(T) = is(Unqual!T == Buffer!U, U);
private enum isThreadgroupMemory(T) = is(Unqual!T == ThreadgroupMemory);
private enum isScalarKernelArgument(T) = isNumeric!(Unqual!T);
private enum isKernelArgument(T) =
    isBuffer!T || isThreadgroupMemory!T || isScalarKernelArgument!T;

struct ThreadgroupMemory {
    NSUInteger length;
}

ThreadgroupMemory threadgroupMemory(NSUInteger length) {
    return ThreadgroupMemory(length);
}

struct CommandBuffer {
    private void* raw_;

    @property MTLCommandBuffer raw() {
        return cast(MTLCommandBuffer) raw_;
    }

    @property void raw(MTLCommandBuffer buf) {
        raw_ = cast(void*) buf;
    }

    void commit() {
        auto buf = raw();
        if (buf !is null) buf.commit();
    }

    void waitUntilCompleted() {
        auto buf = raw();
        if (buf !is null) buf.waitUntilCompleted();
    }

    Encoder computeEncoder() {
        Encoder enc;
        auto buf = raw();
        enc.raw = (buf is null) ? null : buf.computeCommandEncoder();
        return enc;
    }

    void release() {
        auto buf = raw();
        if (buf !is null) buf.release();
        raw_ = null;
    }
}

struct Queue {
    private void* raw_;

    this(Device dev) {
        auto q = dev.raw.newCommandQueue();
        raw_ = cast(void*) q;
    }

    @property MTLCommandQueue raw() {
        return cast(MTLCommandQueue) raw_;
    }

    @property void raw(MTLCommandQueue q) {
        raw_ = cast(void*) q;
    }

    CommandBuffer commandBuffer() {
        CommandBuffer cb;
        auto q = raw();
        cb.raw = (q is null) ? null : q.commandBuffer();
        return cb;
    }

    void release() {
        auto q = raw();
        if (q !is null) q.release();
        raw_ = null;
    }

    void enqueue(Args...)(Pipeline pipeline, MTLSize grid, MTLSize group, Args args)
        if (allSatisfy!(isKernelArgument, Args)) {
        auto cmdBuffer = commandBuffer();
        if (cmdBuffer.raw is null) return;
        auto encoder = cmdBuffer.computeEncoder();
        if (encoder.raw is null) {
            cmdBuffer.release();
            return;
        }
        scope(exit) cmdBuffer.release();
        scope(exit) encoder.release();

        encoder.setPipeline(pipeline);
        foreach (i, ref arg; args) {
            encodeKernelArgument(encoder, arg, cast(NSUInteger) i);
        }
        encoder.dispatchThreads(grid, group);
        encoder.endEncoding();
        cmdBuffer.commit();
        cmdBuffer.waitUntilCompleted();
    }

    /// Batch multiple dispatches into one command buffer (single GPU sync on commit).
    ComputeBatch beginBatch() {
        return ComputeBatch(this);
    }

    auto enqueue(Kernel kernel, MTLSize grid, MTLSize group) {
        static struct Launch {
            Queue q;
            Kernel k;
            MTLSize grid;
            MTLSize group;

            this(Queue q, Kernel k, MTLSize grid, MTLSize group) {
                this.q = q;
                this.k = k;
                this.grid = grid;
                this.group = group;
            }

            void opCall(Args...)(Args args)
                if (allSatisfy!(isKernelArgument, Args)) {
                q.enqueue(k.pipeline, grid, group, args);
            }
        }

        return Launch(this, kernel, grid, group);
    }
}

/// Record several compute dispatches; call commitAndWait() once at the end.
struct ComputeBatch {
    private CommandBuffer cmdBuffer_;
    private Encoder encoder_;
    private bool active_;
    private bool computeEncoderOpen_;

    this(Queue queue) {
        cmdBuffer_ = queue.commandBuffer();
        if (cmdBuffer_.raw is null) {
            active_ = false;
            computeEncoderOpen_ = false;
            return;
        }
        encoder_ = cmdBuffer_.computeEncoder();
        active_ = encoder_.raw !is null;
        computeEncoderOpen_ = active_;
    }

    @property MTLCommandBuffer commandBuffer() {
        return cmdBuffer_.raw;
    }

    /// End the compute encoder so external encoders (e.g. MPS) can use the command buffer.
    void suspendComputeEncoding() {
        if (!active_ || !computeEncoderOpen_) return;
        encoder_.endEncoding();
        encoder_.release();
        computeEncoderOpen_ = false;
    }

    /// Resume compute dispatches after suspendComputeEncoding().
    void resumeComputeEncoding() {
        if (!active_ || computeEncoderOpen_) return;
        encoder_ = cmdBuffer_.computeEncoder();
        computeEncoderOpen_ = encoder_.raw !is null;
    }

    void enqueue(Args...)(Pipeline pipeline, MTLSize grid, MTLSize group, Args args)
        if (allSatisfy!(isKernelArgument, Args)) {
        if (!active_ || !computeEncoderOpen_) return;
        encoder_.setPipeline(pipeline);
        foreach (i, ref arg; args) {
            encodeKernelArgument(encoder_, arg, cast(NSUInteger) i);
        }
        encoder_.dispatchThreads(grid, group);
    }

    void commit() {
        if (!active_) return;
        if (computeEncoderOpen_) {
            encoder_.endEncoding();
            encoder_.release();
            computeEncoderOpen_ = false;
        }
        cmdBuffer_.commit();
    }

    void wait() {
        if (cmdBuffer_.raw is null) {
            active_ = false;
            return;
        }
        cmdBuffer_.waitUntilCompleted();
        cmdBuffer_.release();
        active_ = false;
    }

    void commitAndWait() {
        commit();
        wait();
    }
}

private void encodeKernelArgument(T)(ref Encoder encoder, ref T buffer, NSUInteger index)
    if (isBuffer!T) {
    encoder.setBuffer(buffer, 0, index);
}

private void encodeKernelArgument(T)(ref Encoder encoder, ref T memory, NSUInteger index)
    if (isThreadgroupMemory!T) {
    encoder.setThreadgroupMemoryLength(memory.length, index);
}

private void encodeKernelArgument(T)(ref Encoder encoder, ref T value, NSUInteger index)
    if (isScalarKernelArgument!T) {
    encoder.setBytes(&value, T.sizeof, index);
}
