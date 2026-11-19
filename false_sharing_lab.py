
import os
import sys
import json
import time
import hashlib
import platform
import threading
import subprocess

import numpy as np
import psutil
from numba import njit


# ============================================================
# STUDENT CONFIGURATION
# ============================================================

GIT_USERNAME = "belkamenbirgee"
NUM_THREADS = 4
ITERATIONS_PER_THREAD = 60_000_000
CACHE_LINE_SIZE = 64


# ============================================================
# TASK 1: HARDWARE TOPOLOGY AND STUDENT IDENTITY
# ============================================================

def detect_cache_line_size():

    linux_path = (
        "/sys/devices/system/cpu/cpu0/cache/"
        "index0/coherency_line_size"
    )

    if os.path.exists(linux_path):
        with open(linux_path, "r", encoding="utf-8") as file:
            return int(file.read().strip())

    if sys.platform == "darwin":
        result = subprocess.check_output(
            ["sysctl", "-n", "hw.cachelinesize"],
            encoding="utf-8"
        )
        return int(result.strip())

    return CACHE_LINE_SIZE


def init_system_context(git_user):

    logical_cores = os.cpu_count() or 1

    if logical_cores < NUM_THREADS:
        raise RuntimeError("At least four logical cores are required.")

    line_size = detect_cache_line_size()
    cpu_model = platform.processor()

    cpu_signature = (
        f"{platform.machine()}::"
        f"{cpu_model}::"
        f"{logical_cores}"
    )

    seed_input = f"{git_user}::{cpu_signature}".encode("utf-8")
    seed_hash = hashlib.sha256(seed_input).hexdigest()
    instance_seed = int(seed_hash[:16], 16)

    available_cores = list(range(logical_cores))
    affinity_supported = False

    try:
        available_cores = sorted(psutil.Process().cpu_affinity())
        affinity_supported = True
    except (OSError, AttributeError):
        pass

    if len(available_cores) < NUM_THREADS:
        raise RuntimeError("Fewer than four CPU cores are available.")

    selected_cores = available_cores[:NUM_THREADS]

    try:
        psutil.Process().cpu_affinity(selected_cores)
    except (OSError, AttributeError):
        affinity_supported = False

    return {
        "git_username": git_user,
        "cpu_model": cpu_model,
        "architecture": platform.machine(),
        "logical_cores": logical_cores,
        "cache_line_size_bytes": line_size,
        "elements_per_cache_line": line_size // 8,
        "instance_seed": instance_seed,
        "available_cores": selected_cores,
        "affinity_supported": affinity_supported
    }


# ============================================================
# TASK 2: GIL-FREE NATIVE COUNTER KERNELS
# ============================================================

@njit(nogil=True)
def worker_strided_accumulate(buffer, slot, iters, thread_seed):

    for i in range(iters):
        buffer[slot] = buffer[slot] + np.uint64(1)


@njit(nogil=True)
def worker_private_accumulate(buffer, slot, iters, thread_seed):

    local_counter = np.uint64(0)

    for i in range(iters):
        local_counter += np.uint64(1)

    buffer[slot] = local_counter


# ============================================================
# ALIGNED MEMORY ALLOCATION
# ============================================================

def allocate_aligned_uint64(count, alignment=64):

    raw = np.zeros(
        count * 8 + alignment,
        dtype=np.uint8
    )

    offset = (-raw.ctypes.data) % alignment

    aligned = raw[offset:offset + count * 8].view(np.uint64)

    assert aligned.ctypes.data % alignment == 0

    return aligned


# ============================================================
# TASK 2 TEST
# ============================================================

def test_task2():

    buffer = allocate_aligned_uint64(16)

    address = buffer.ctypes.data

    print("Array address:", address)
    print("64-byte aligned:", address % 64 == 0)

    assert address % 64 == 0

    worker_strided_accumulate(buffer, 0, 1000, 123)

    print("Counter after 1000 updates:", buffer[0])

    assert buffer[0] == 1000

    print("Task 2 kernel test PASSED!")


# ============================================================
# TASK 3: CONTEXT SWITCH MEASUREMENT
# ============================================================

def get_context_switches():

    switches = psutil.Process().num_ctx_switches()

    return {
        "voluntary": switches.voluntary,
        "involuntary": switches.involuntary
    }


# ============================================================
# TASK 3: STRIDE BENCHMARK
# ============================================================

def run_stride_benchmark(context):

    results = {}

    for stride in (1, 4, 8, 16):

        print(f"\nStarting stride={stride} benchmark...")

        buffer = allocate_aligned_uint64(
            (NUM_THREADS - 1) * stride + 1
        )

        threads = []

        def thread_job(thread_id):

            slot = thread_id * stride

            thread_seed = (
                context["instance_seed"] + thread_id
            ) & 0xFFFFFFFFFFFFFFFF

            worker_strided_accumulate(
                buffer,
                slot,
                ITERATIONS_PER_THREAD,
                thread_seed
            )

        switches_before = get_context_switches()
        start_ns = time.perf_counter_ns()

        for thread_id in range(NUM_THREADS):

            thread = threading.Thread(
                target=thread_job,
                args=(thread_id,)
            )

            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        elapsed_ns = time.perf_counter_ns() - start_ns
        switches_after = get_context_switches()

        expected = np.uint64(ITERATIONS_PER_THREAD)

        for thread_id in range(NUM_THREADS):

            slot = thread_id * stride

            assert buffer[slot] == expected, (
                f"Incorrect counter: stride={stride}, "
                f"thread={thread_id}"
            )

        total_updates = NUM_THREADS * ITERATIONS_PER_THREAD

        results[str(stride)] = {
            "elapsed_ns": elapsed_ns,
            "total_updates": total_updates,
            "ns_per_update": elapsed_ns / total_updates,
            "voluntary_context_switches": (
                switches_after["voluntary"]
                - switches_before["voluntary"]
            ),
            "involuntary_context_switches": (
                switches_after["involuntary"]
                - switches_before["involuntary"]
            ),
            "final_counters": [
                int(buffer[i * stride])
                for i in range(NUM_THREADS)
            ]
        }

        print(f"Stride: {stride}")
        print(f"Elapsed: {elapsed_ns:,} ns")
        print(
            f"Nanoseconds per update: "
            f"{elapsed_ns / total_updates:.6f}"
        )
        print("Final counters:", results[str(stride)]["final_counters"])

    return results


# ============================================================
# TASK 4: PRIVATE COUNTER BASELINE
# ============================================================

def run_private_benchmark(context):

    print("\nStarting private counter benchmark...")

    buffer = allocate_aligned_uint64(NUM_THREADS)
    threads = []

    def thread_job(thread_id):

        thread_seed = (
            context["instance_seed"] + thread_id
        ) & 0xFFFFFFFFFFFFFFFF

        worker_private_accumulate(
            buffer,
            thread_id,
            ITERATIONS_PER_THREAD,
            thread_seed
        )

    switches_before = get_context_switches()
    start_ns = time.perf_counter_ns()

    for thread_id in range(NUM_THREADS):

        thread = threading.Thread(
            target=thread_job,
            args=(thread_id,)
        )

        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    elapsed_ns = time.perf_counter_ns() - start_ns
    switches_after = get_context_switches()

    expected = np.uint64(ITERATIONS_PER_THREAD)

    for thread_id in range(NUM_THREADS):
        assert buffer[thread_id] == expected

    total_updates = NUM_THREADS * ITERATIONS_PER_THREAD

    results = {
        "elapsed_ns": elapsed_ns,
        "total_updates": total_updates,
        "ns_per_update": elapsed_ns / total_updates,
        "voluntary_context_switches": (
            switches_after["voluntary"]
            - switches_before["voluntary"]
        ),
        "involuntary_context_switches": (
            switches_after["involuntary"]
            - switches_before["involuntary"]
        ),
        "final_counters": [int(value) for value in buffer]
    }

    print(f"Private elapsed: {elapsed_ns:,} ns")
    print(
        f"Nanoseconds per update: "
        f"{elapsed_ns / total_updates:.6f}"
    )
    print("Final counters:", results["final_counters"])

    return results


# ============================================================
# TASK 5: LLVM INSPECTION
# ============================================================

def inspect_worker_llvm():

    print("\n========== LLVM INSPECTION ==========")

    if not worker_strided_accumulate.signatures:
        raise RuntimeError(
            "Worker has not been compiled. Run Task 2 first."
        )

    signature = worker_strided_accumulate.signatures[0]
    llvm_ir = worker_strided_accumulate.inspect_llvm(signature)

    print("LLVM IR length:", len(llvm_ir))
    print("LLVM load instructions:", llvm_ir.count("load i64"))
    print("LLVM store instructions:", llvm_ir.count("store i64"))
    print("LLVM branch instructions:", llvm_ir.count("br label"))

    # Save the LLVM output for manual inspection.
    with open("worker_llvm.txt", "w", encoding="utf-8") as file:
        file.write(llvm_ir)

    # This is only a preliminary check.
    # We must inspect whether the memory load/store is inside the loop.
    print("LLVM output saved to worker_llvm.txt")

    return llvm_ir


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    context = init_system_context(GIT_USERNAME)

    print("========== SYSTEM CONTEXT ==========")
    print(json.dumps(context, indent=4))

    print("\n========== TASK 2 TEST ==========")
    test_task2()

    print("\n========== TASK 3 BENCHMARK ==========")
    stride_results = run_stride_benchmark(context)

    print("\n========== STRIDE RESULTS ==========")
    print(json.dumps(stride_results, indent=4))

    print("\n========== TASK 4 PRIVATE BASELINE ==========")
    private_results = run_private_benchmark(context)

    print("\n========== PRIVATE RESULTS ==========")
    print(json.dumps(private_results, indent=4))

    print("\n========== TASK 5 LLVM INSPECTION ==========")
    inspect_worker_llvm()

    print("\n========== ALL BENCHMARKS COMPLETED ==========")
