import os
import time
import argparse
import statistics

import torch
import torch.distributed as dist


def sync():
    torch.cuda.synchronize()
    dist.barrier()


def now():
    return time.perf_counter()


def print0(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs)


def get_device_info():
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.cuda.current_device()
    name = torch.cuda.get_device_name(device)
    print(f"[rank {rank}] local_rank={local_rank}, cuda_device={device}, name={name}", flush=True)


def sp_to_head(input_, group):
    """
    scatter_dim == 2 and gather_dim == 1
    input:  [bs, shard_seqlen, hn, hd]
    output: [bs, seqlen, shard_hn, hd]
    """
    world_size = dist.get_world_size(group)
    bs, shard_seqlen, hn, hd = input_.shape
    seqlen = shard_seqlen * world_size
    shard_hn = hn // world_size

    # pack
    t0 = now()
    input_ = input_.transpose(0, 2).contiguous()   # [hn, shard_seqlen, bs, hd]
    sync()
    t1 = now()

    # comm
    output = torch.empty_like(input_)
    dist.all_to_all_single(output, input_, group=group)
    sync()
    t2 = now()

    # unpack
    # 等价于 cat(split(...), dim=1) 的更规整写法
    output = output.reshape(world_size, shard_hn, shard_seqlen, bs, hd)
    output = output.permute(1, 0, 2, 3, 4).reshape(shard_hn, seqlen, bs, hd)
    output = output.transpose(0, 2).contiguous()   # [bs, seqlen, shard_hn, hd]
    sync()
    t3 = now()

    return output, (t1 - t0), (t2 - t1), (t3 - t2)


def head_to_sp(input_, group):
    """
    scatter_dim == 1 and gather_dim == 2
    input:  [bs, seqlen, shard_hn, hd]
    output: [bs, shard_seqlen, hn, hd]
    """
    world_size = dist.get_world_size(group)
    bs, seqlen, shard_hn, hd = input_.shape
    hn = shard_hn * world_size
    shard_seqlen = seqlen // world_size

    # pack
    t0 = now()
    input_ = input_.transpose(0, 2).contiguous()  # [shard_hn, seqlen, bs, hd]
    input_ = (
        input_
        .reshape(shard_hn, world_size, shard_seqlen, bs, hd)
        .transpose(0, 1)
        .reshape(shard_hn * world_size, shard_seqlen, bs, hd)
        .contiguous()
    )
    sync()
    t1 = now()

    # comm
    output = torch.empty_like(input_)
    dist.all_to_all_single(output, input_, group=group)
    sync()
    t2 = now()

    # unpack
    output = output.transpose(0, 2).contiguous()  # [bs, shard_seqlen, hn, hd]
    sync()
    t3 = now()

    return output, (t1 - t0), (t2 - t1), (t3 - t2)


def run_bench(fn, input_, group, warmup, iters, tag):
    pack_ts = []
    comm_ts = []
    unpack_ts = []
    total_ts = []

    for _ in range(warmup):
        _ = fn(input_, group)

    sync()

    for _ in range(iters):
        sync()
        t0 = now()
        _, t_pack, t_comm, t_unpack = fn(input_, group)
        sync()
        t1 = now()

        pack_ts.append(t_pack * 1000)
        comm_ts.append(t_comm * 1000)
        unpack_ts.append(t_unpack * 1000)
        total_ts.append((t1 - t0) * 1000)

    def stat(xs):
        return statistics.mean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0

    pack_m, pack_s = stat(pack_ts)
    comm_m, comm_s = stat(comm_ts)
    unpack_m, unpack_s = stat(unpack_ts)
    total_m, total_s = stat(total_ts)

    print0(f"\n[{tag}]")
    print0(f"pack   : {pack_m:.3f} ms ± {pack_s:.3f}")
    print0(f"comm   : {comm_m:.3f} ms ± {comm_s:.3f}")
    print0(f"unpack : {unpack_m:.3f} ms ± {unpack_s:.3f}")
    print0(f"total  : {total_m:.3f} ms ± {total_s:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=2)
    parser.add_argument("--seqlen", type=int, default=4680)
    parser.add_argument("--hn", type=int, default=12)
    parser.add_argument("--hd", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # assert args.seqlen % world_size == 0, "seqlen must be divisible by world_size"
    # assert args.hn % world_size == 0, "hn must be divisible by world_size"

    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    get_device_info()

    shard_seqlen = args.seqlen // world_size
    shard_hn = args.hn // world_size

    # 模拟 SP -> head 的输入: [bs, shard_seqlen, hn, hd]
    sp_input = torch.randn(
        args.bs, shard_seqlen, args.hn, args.hd,
        device=device, dtype=dtype
    )

    # 模拟 head -> SP 的输入: [bs, seqlen, shard_hn, hd]
    head_input = torch.randn(
        args.bs, args.seqlen, shard_hn, args.hd,
        device=device, dtype=dtype
    )

    print0(
        f"world_size={world_size}, bs={args.bs}, seqlen={args.seqlen}, "
        f"hn={args.hn}, hd={args.hd}, dtype={args.dtype}"
    )
    print0(f"sp_input   shape = {tuple(sp_input.shape)}")
    print0(f"head_input shape = {tuple(head_input.shape)}")

    run_bench(sp_to_head, sp_input, dist.group.WORLD, args.warmup, args.iters, "SP -> HEAD")
    run_bench(head_to_sp, head_input, dist.group.WORLD, args.warmup, args.iters, "HEAD -> SP")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()