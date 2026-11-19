
import os
import sys
import json
import math
import hashlib
import subprocess
import numpy as np

from mpi4py import MPI
from numba import cuda, float64

# ============================================================
# GLOBAL CONSTANTS
# ============================================================

TILE_DIM = 16
SHARED_DIM = TILE_DIM + 2

GLOBAL_NX = 512
GLOBAL_NY = 512

LOCAL_NX = 256
LOCAL_NY = 256

STEPS = 400
LAMBDA_DAMP = 0.15

DX = 1.0 / 513
DY = 1.0 / 513

# ============================================================
# GPU UUID
# ============================================================

def get_gpu_uuid():
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=uuid",
            "--format=csv,noheader"
        ],
        encoding="utf-8"
    )
    return out.strip().splitlines()[0].strip()


# ============================================================
# STUDENT SEED
# ============================================================

def derive_student_seed(github_user, gpu_uuid):
    digest = hashlib.sha256(
        f"{github_user.strip()}::{gpu_uuid}".encode("utf-8")
    ).hexdigest()

    return int(digest[:16], 16)


# ============================================================
# TASK 1: MPI DISTRIBUTED TOPOLOGY
# ============================================================

def init_distributed_context(github_user):

    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()

    if size != 4:
        raise RuntimeError(
            f"Expected exactly 4 MPI ranks, got {size}"
        )

    # Create a 2 x 2 Cartesian process grid
    dims = [2, 2]
    periods = [False, False]

    cart_comm = comm.Create_cart(
        dims=dims,
        periods=periods,
        reorder=False
    )

    # Get Cartesian coordinates
    coords = cart_comm.Get_coords(rank)
    row_rank, col_rank = coords

    # Find neighboring ranks
    nbr_up, nbr_down = cart_comm.Shift(0, 1)
    nbr_left, nbr_right = cart_comm.Shift(1, 1)

    # Select GPU 0
    cuda.select_device(0)

    # Get GPU UUID on rank 0 and share it
    gpu_uuid = get_gpu_uuid() if rank == 0 else None
    gpu_uuid = cart_comm.bcast(gpu_uuid, root=0)

    # Generate hardware-bound seed
    instance_seed = derive_student_seed(
        github_user,
        gpu_uuid
    )

    print(
        f"Rank {rank}: coordinates={coords}, "
        f"neighbors={(nbr_up, nbr_down, nbr_left, nbr_right)}"
    )

    return (
        cart_comm,
        rank,
        size,
        row_rank,
        col_rank,
        nbr_up,
        nbr_down,
        nbr_left,
        nbr_right,
        instance_seed
    )

# ============================================================
# TASK 2: SEEDED HETEROGENEOUS SUBDOMAIN INITIALIZATION
# ============================================================

def initialize_subdomains(
    row_rank,
    col_rank,
    instance_seed,
    nbr_up,
    nbr_down,
    nbr_left,
    nbr_right
):

    # Each MPI rank owns 256 x 256 cells
    ny = LOCAL_NY
    nx = LOCAL_NX

    # Local padded coordinates: 0..257
    local_y = np.arange(ny + 2)
    local_x = np.arange(nx + 2)

    # Convert local indices to global physical coordinates
    global_y = row_rank * ny + local_y - 1
    global_x = col_rank * nx + local_x - 1

    y = (global_y + 1) * DY
    x = (global_x + 1) * DX

    X, Y = np.meshgrid(x, y)

    # Rank-specific deterministic random generator
    rng = np.random.default_rng(
        (int(instance_seed) + row_rank * 2 + col_rank) % (2**64)
    )

    # --------------------------------------------------------
    # Spatially heterogeneous diffusivity: 0.08 to 0.24
    # --------------------------------------------------------

    kappa = (
        0.08
        + 0.16 * (
            0.5
            + 0.5 * np.sin(2 * np.pi * X)
            * np.cos(2 * np.pi * Y)
        )
    )

    kappa = np.clip(kappa, 0.08, 0.24)
    kappa = kappa.astype(np.float64)

    # --------------------------------------------------------
    # Initial temperature: seeded Gaussian thermal pulses
    # --------------------------------------------------------

    u = np.zeros((ny + 2, nx + 2), dtype=np.float64)

    number_of_pulses = 3

    for pulse in range(number_of_pulses):

        center_x = rng.uniform(0.2, 0.8)
        center_y = rng.uniform(0.2, 0.8)

        amplitude = rng.uniform(0.5, 1.0)

        sigma = rng.uniform(0.04, 0.10)

        gaussian = amplitude * np.exp(
            -(
                (X - center_x) ** 2
                + (Y - center_y) ** 2
            ) / (2 * sigma ** 2)
        )

        u += gaussian

    # --------------------------------------------------------
    # Localized source field
    # --------------------------------------------------------

    source_x = 0.5
    source_y = 0.5
    source_sigma = 0.08

    source = np.exp(
        -(
            (X - source_x) ** 2
            + (Y - source_y) ** 2
        ) / (2 * source_sigma ** 2)
    )

    source = source.astype(np.float64)

    # --------------------------------------------------------
    # Enforce physical outer boundary conditions
    # --------------------------------------------------------

    if nbr_up == MPI.PROC_NULL:
        u[0, :] = 0.0

    if nbr_down == MPI.PROC_NULL:
        u[ny + 1, :] = 0.0

    if nbr_left == MPI.PROC_NULL:
        u[:, 0] = 0.0

    if nbr_right == MPI.PROC_NULL:
        u[:, nx + 1] = 0.0

    return u, kappa, source


# ============================================================
# TASK 3: GPU HALO PACK / UNPACK KERNELS
# ============================================================

@cuda.jit
def pack_halos_kernel(u, send_buffer, direction, nx, ny):

    idx = cuda.grid(1)

    if idx < max(nx, ny):

        if direction == 0 and idx < nx:
            send_buffer[idx] = u[1, idx + 1]

        elif direction == 1 and idx < nx:
            send_buffer[idx] = u[ny, idx + 1]

        elif direction == 2 and idx < ny:
            send_buffer[idx] = u[idx + 1, 1]

        elif direction == 3 and idx < ny:
            send_buffer[idx] = u[idx + 1, nx]


@cuda.jit
def unpack_halos_kernel(u, recv_buffer, direction, nx, ny):

    idx = cuda.grid(1)

    if idx < max(nx, ny):

        if direction == 0 and idx < nx:
            u[0, idx + 1] = recv_buffer[idx]

        elif direction == 1 and idx < nx:
            u[ny + 1, idx + 1] = recv_buffer[idx]

        elif direction == 2 and idx < ny:
            u[idx + 1, 0] = recv_buffer[idx]

        elif direction == 3 and idx < ny:
            u[idx + 1, nx + 1] = recv_buffer[idx]


# ============================================================
# TASK 3: ASYNCHRONOUS MPI HALO EXCHANGE
# ============================================================

def exchange_halos_async(
    u,
    cart_comm,
    nbr_up,
    nbr_down,
    nbr_left,
    nbr_right,
    stream_boundary
):

    nx = LOCAL_NX
    ny = LOCAL_NY

    neighbors = [
        nbr_up,
        nbr_down,
        nbr_left,
        nbr_right
    ]

    # Directions: 0=up, 1=down, 2=left, 3=right
    lengths = [nx, nx, ny, ny]

    # Opposite-direction message tags
    send_tags = [0, 1, 2, 3]
    recv_tags = [1, 0, 3, 2]

    max_length = max(nx, ny)

    # Allocate pinned host buffers
    host_send = [
        cuda.pinned_array(max_length, dtype=np.float64)
        for _ in range(4)
    ]

    host_recv = [
        cuda.pinned_array(max_length, dtype=np.float64)
        for _ in range(4)
    ]

    # Allocate GPU buffers
    device_send = [
        cuda.device_array(max_length, dtype=np.float64)
        for _ in range(4)
    ]

    device_recv = [
        cuda.device_array(max_length, dtype=np.float64)
        for _ in range(4)
    ]

    # --------------------------------------------------------
    # Stage A: GPU packing and asynchronous D2H transfers
    # --------------------------------------------------------

    threads = 256
    blocks = (max_length + threads - 1) // threads

    for direction in range(4):

        pack_halos_kernel[blocks, threads, stream_boundary](
            u,
            device_send[direction],
            direction,
            nx,
            ny
        )

        device_send[direction].copy_to_host(
            host_send[direction],
            stream=stream_boundary
        )

    stream_boundary.synchronize()

    # Initialize receive buffers to zero.
    # This preserves zero values for MPI.PROC_NULL neighbors.
    for direction in range(4):
        host_recv[direction][:] = 0.0

    # --------------------------------------------------------
    # Stage B: Non-blocking MPI communication
    # --------------------------------------------------------

    requests = []

    # Post receives first
    for direction in range(4):

        request = cart_comm.Irecv(
            host_recv[direction][:lengths[direction]],
            source=neighbors[direction],
            tag=recv_tags[direction]
        )

        requests.append(request)

    # Post sends
    for direction in range(4):

        request = cart_comm.Isend(
            host_send[direction][:lengths[direction]],
            dest=neighbors[direction],
            tag=send_tags[direction]
        )

        requests.append(request)

    MPI.Request.Waitall(requests)

    # --------------------------------------------------------
    # Stage C: Asynchronous H2D transfers and unpacking
    # --------------------------------------------------------

    for direction in range(4):

        device_recv[direction].copy_to_device(
            host_recv[direction],
            stream=stream_boundary
        )

        unpack_halos_kernel[blocks, threads, stream_boundary](
            u,
            device_recv[direction],
            direction,
            nx,
            ny
        )

    stream_boundary.synchronize()

    return host_send, host_recv

# ============================================================

# ============================================================
# TASK 4: SHARED-MEMORY TILED CUDA STENCIL
# ============================================================

@cuda.jit
def tiled_stencil_kernel(u, kappa, source, out, dt,
                         boundary_only):
    tile = cuda.shared.array((18, 18), dtype=float64)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    tid = ty * 16 + tx

    base_r = cuda.blockIdx.y * 16
    base_c = cuda.blockIdx.x * 16

    # Cooperatively load all 324 values into shared memory.
    # 256 threads load one value each; first 68 load a second.
    for item in range(2):
        index = tid + item * 256

        if index < 324:
            sr = index // 18
            sc = index % 18

            tile[sr, sc] = u[base_r + sr, base_c + sc]

    cuda.syncthreads()

    r = base_r + ty + 1
    c = base_c + tx + 1

    if r <= 256 and c <= 256:

        is_deep = (
            r >= 2 and r <= 255
            and c >= 2 and c <= 255
        )

        # 0 = full interior, 1 = deep interior, 2 = boundary ring.
        should_compute = (
            boundary_only == 0
            or (boundary_only == 1 and is_deep)
            or (boundary_only == 2 and not is_deep)
        )

        if should_compute:

            center = tile[ty + 1, tx + 1]
            north = tile[ty, tx + 1]
            south = tile[ty + 2, tx + 1]
            west = tile[ty + 1, tx]
            east = tile[ty + 1, tx + 2]

            kc = kappa[r, c]

            kn = 0.5 * (kc + kappa[r - 1, c])
            ks = 0.5 * (kc + kappa[r + 1, c])
            kw = 0.5 * (kc + kappa[r, c - 1])
            ke = 0.5 * (kc + kappa[r, c + 1])

            dx = 1.0 / 513.0
            dy = 1.0 / 513.0

            diffusion = (
                (ke * (east - center)
                 - kw * (center - west)) / (dx * dx)
                +
                (ks * (south - center)
                 - kn * (center - north)) / (dy * dy)
            )

            reaction = -0.15 * center * center * center

            out[r, c] = center + dt * (
                diffusion + reaction + source[r, c]
            )


# ============================================================
# TASK 4: ASYNCHRONOUS HALO EXCHANGE
# ============================================================

def begin_halo_exchange(u, cart_comm,
                        nbr_up, nbr_down,
                        nbr_left, nbr_right,
                        stream_boundary):

    nx = LOCAL_NX
    ny = LOCAL_NY

    neighbors = [
        nbr_up, nbr_down, nbr_left, nbr_right
    ]

    lengths = [nx, nx, ny, ny]
    send_tags = [0, 1, 2, 3]
    recv_tags = [1, 0, 3, 2]

    max_length = max(nx, ny)
    threads = 256
    blocks = (max_length + threads - 1) // threads

    host_send = [
        cuda.pinned_array(max_length, dtype=np.float64)
        for _ in range(4)
    ]

    host_recv = [
        cuda.pinned_array(max_length, dtype=np.float64)
        for _ in range(4)
    ]

    device_send = [
        cuda.device_array(max_length, dtype=np.float64)
        for _ in range(4)
    ]

    device_recv = [
        cuda.device_array(max_length, dtype=np.float64)
        for _ in range(4)
    ]

    # GPU packing and D2H transfer.
    for direction in range(4):
        pack_halos_kernel[blocks, threads, stream_boundary](
            u, device_send[direction], direction, nx, ny
        )

        device_send[direction].copy_to_host(
            host_send[direction],
            stream=stream_boundary
        )

    # MPI must not read the send buffers until D2H is complete.
    stream_boundary.synchronize()

    for direction in range(4):
        host_recv[direction][:] = 0.0

    requests = []

    # Post all receives first.
    for direction in range(4):
        requests.append(
            cart_comm.Irecv(
                host_recv[direction][:lengths[direction]],
                source=neighbors[direction],
                tag=recv_tags[direction]
            )
        )

    # Then post all sends.
    for direction in range(4):
        requests.append(
            cart_comm.Isend(
                host_send[direction][:lengths[direction]],
                dest=neighbors[direction],
                tag=send_tags[direction]
            )
        )

    # Do NOT wait here. The caller computes the deep interior.

    return {
        "requests": requests,
        "host_send": host_send,
        "host_recv": host_recv,
        "device_recv": device_recv,
        "lengths": lengths,
        "blocks": blocks,
        "threads": threads,
        "u": u
    }


def finish_halo_exchange(exchange, interior_stream,
                         stream_boundary):

    # Wait for MPI while deep-interior CUDA work can run.
    MPI.Request.Waitall(exchange["requests"])

    # Ensure deep computation has finished before ghost updates.
    interior_stream.synchronize()

    u = exchange["u"]

    # Copy received halos to GPU and unpack.
    for direction in range(4):
        exchange["device_recv"][direction].copy_to_device(
            exchange["host_recv"][direction],
            stream=stream_boundary
        )

        unpack_halos_kernel[
            exchange["blocks"],
            exchange["threads"],
            stream_boundary
        ](
            u,
            exchange["device_recv"][direction],
            direction,
            LOCAL_NX,
            LOCAL_NY
        )

    stream_boundary.synchronize()


# ============================================================
# TASK 5: METRICS AND PTX AUDIT
# ============================================================

def calculate_metrics(u, previous_u):
    interior = u[1:-1, 1:-1]
    previous = previous_u[1:-1, 1:-1]

    energy = float(np.sum(interior) * DX * DY)

    residual = float(
        np.sqrt(np.sum((interior - previous) ** 2))
    )

    return energy, residual


def global_metrics(cart_comm, d_u, d_previous):
    u_host = d_u.copy_to_host()
    previous_host = d_previous.copy_to_host()

    local_energy, local_residual = calculate_metrics(
        u_host, previous_host
    )

    energy = cart_comm.allreduce(
        local_energy, op=MPI.SUM
    )

    residual_squared = cart_comm.allreduce(
        local_residual ** 2, op=MPI.SUM
    )

    return energy, float(np.sqrt(residual_squared))


def ptx_audit():
    compiled = tiled_stencil_kernel.inspect_asm()

    if isinstance(compiled, dict):
        ptx = next(iter(compiled.values()))
    else:
        ptx = compiled

    return {
        "shared_loads": ptx.count("ld.shared"),
        "shared_stores": ptx.count("st.shared"),
        "bar_sync": ptx.count("bar.sync"),
        "has_shared_memory": (
            "ld.shared" in ptx and "st.shared" in ptx
        ),
        "has_barrier": "bar.sync" in ptx
    }


# ============================================================
# TASK 5: ONE SERIALIZED OR OVERLAPPED SIMULATION
# ============================================================

def run_simulation(initial_u, kappa, source,
                   cart_comm, neighbors, dt,
                   mode, rank):

    d_u = cuda.to_device(initial_u)
    d_next = cuda.device_array_like(d_u)
    d_previous = cuda.device_array_like(d_u)

    stream_boundary = cuda.stream()
    stream_interior = cuda.stream()

    nbr_up, nbr_down, nbr_left, nbr_right = neighbors

    blocks = (16, 16)
    threads = (16, 16)

    d_kappa = cuda.to_device(kappa)
    d_source = cuda.to_device(source)

    energy_trace = []
    residual_trace = []

    # Initial metric at step 0.
    d_previous.copy_to_device(d_u)
    energy, residual = global_metrics(
        cart_comm, d_u, d_previous
    )

    if rank == 0:
        energy_trace.append((0, energy))
        residual_trace.append((0, residual))

    cart_comm.Barrier()
    start_time = MPI.Wtime()

    for step in range(1, STEPS + 1):

        # Preserve the previous state for residual calculation.
        d_previous.copy_to_device(d_u)

        # Keep ghost cells from the current state.
        d_next.copy_to_device(d_u)

        if mode == "serialized":

            # Complete halo exchange before stencil computation.
            exchange = begin_halo_exchange(
                d_u, cart_comm,
                nbr_up, nbr_down,
                nbr_left, nbr_right,
                stream_boundary
            )

            finish_halo_exchange(
                exchange, stream_interior, stream_boundary
            )

            tiled_stencil_kernel[
                blocks, threads, stream_boundary
            ](
                d_u, d_kappa, d_source, d_next,
                dt, 0
            )

            stream_boundary.synchronize()

        else:

            # Start nonblocking halo communication.
            exchange = begin_halo_exchange(
                d_u, cart_comm,
                nbr_up, nbr_down,
                nbr_left, nbr_right,
                stream_boundary
            )

            # Compute only the deep interior while MPI is active.
            tiled_stencil_kernel[
                blocks, threads, stream_interior
            ](
                d_u, d_kappa, d_source, d_next,
                dt, 1
            )

            # Finish communication and unpack the ghost cells.
            finish_halo_exchange(
                exchange, stream_interior, stream_boundary
            )

            # Compute only the boundary ring.
            tiled_stencil_kernel[
                blocks, threads, stream_boundary
            ](
                d_u, d_kappa, d_source, d_next,
                dt, 2
            )

            stream_boundary.synchronize()

        d_u, d_next = d_next, d_u

        # Global metrics every 100 steps.
        if step % 100 == 0:
            energy, residual = global_metrics(
                cart_comm, d_u, d_previous
            )

            if rank == 0:
                energy_trace.append((step, energy))
                residual_trace.append((step, residual))

                print(
                    f"{mode} | Step {step:3d} | "
                    f"Energy={energy:.12e} | "
                    f"L2 residual={residual:.12e}"
                )

    stream_boundary.synchronize()
    stream_interior.synchronize()

    elapsed = MPI.Wtime() - start_time
    elapsed = cart_comm.allreduce(elapsed, op=MPI.MAX)

    final_u = d_u.copy_to_host()

    return final_u, elapsed, energy_trace, residual_trace


# ============================================================
# TASK 5: COMPLETE SOLVER
# ============================================================

def run_solver(github_user):

    (
        cart_comm, rank, size,
        row_rank, col_rank,
        nbr_up, nbr_down, nbr_left, nbr_right,
        instance_seed
    ) = init_distributed_context(github_user)

    u, kappa, source = initialize_subdomains(
        row_rank, col_rank, instance_seed,
        nbr_up, nbr_down, nbr_left, nbr_right
    )

    # Use a globally consistent maximum diffusivity.
    local_kappa_max = float(np.max(kappa))
    kappa_max = cart_comm.allreduce(
        local_kappa_max, op=MPI.MAX
    )

    dt_max = (DX ** 2) / (4.0 * kappa_max)
    dt = 0.2 * dt_max

    if rank == 0:
        print("\\nStarting distributed CUDA stencil solver")
        print("MPI ranks:", size)
        print("Global grid:", GLOBAL_NX, "x", GLOBAL_NY)
        print("Local grid:", LOCAL_NX, "x", LOCAL_NY)
        print("Steps:", STEPS)
        print("Maximum diffusivity:", kappa_max)
        print("dt:", dt)

    neighbors = (
        nbr_up, nbr_down, nbr_left, nbr_right
    )

    initial_u = u.copy()

    # Reference run: serialized communication and computation.
    serialized, serial_time, serial_energy, serial_residual = (
        run_simulation(
            initial_u, kappa, source,
            cart_comm, neighbors, dt,
            "serialized", rank
        )
    )

    cart_comm.Barrier()

    # Test run: communication overlaps deep-interior computation.
    overlapped, overlap_time, overlap_energy, overlap_residual = (
        run_simulation(
            initial_u, kappa, source,
            cart_comm, neighbors, dt,
            "overlapped", rank
        )
    )

    # Compare final states across all ranks.
    local_max_diff = float(
        np.max(np.abs(serialized - overlapped))
    )

    max_diff = cart_comm.allreduce(
        local_max_diff, op=MPI.MAX
    )

    audit = ptx_audit()

    if rank == 0:
        print("\\n========== FINAL RESULTS ==========")
        print("Serialized time:", serial_time)
        print("Overlapped time:", overlap_time)
        print("Maximum final-state difference:", max_diff)
        print("Numerical comparison (< 1e-12):", max_diff < 1e-12)
        print("PTX audit:", audit)

        manifest = {
            "github_username": github_user,
            "instance_seed": int(instance_seed),
            "gpu_uuid": get_gpu_uuid(),
            "mpi_ranks": int(size),
            "global_grid": [GLOBAL_NX, GLOBAL_NY],
            "local_grid": [LOCAL_NX, LOCAL_NY],
            "steps": STEPS,
            "dt": float(dt),
            "kappa_max": float(kappa_max),
            "serialized_time_seconds": float(serial_time),
            "overlapped_time_seconds": float(overlap_time),
            "max_final_state_difference": float(max_diff),
            "numerical_match_1e_12": bool(max_diff < 1e-12),
            "ptx_audit": audit,
            "energy_trace": serial_energy
        }

        with open("submission_manifest.json", "w") as f:
            json.dump(manifest, f, indent=4)

        print("\\nCreated submission_manifest.json")

    cart_comm.Barrier()
    cart_comm.Free()

    return max_diff


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise RuntimeError(
            "Usage: python distributed_cuda_solver.py GITHUB_USERNAME"
        )

    run_solver(sys.argv[1])
