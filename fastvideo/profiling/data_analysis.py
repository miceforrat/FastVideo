from collections import defaultdict

def explore_node_dict(root: dict):
    stats = defaultdict(lambda: {
        "durations": [],
        "input_shapes": [],
        "output_shapes": [],
        "meta": []
    })
    def dfs(node: dict, path: tuple[str, ...]):
        name = node["name"]
        cur_path = path + (name,)
        key = "->".join(cur_path)

        stats[key]["durations"].append(node.get("duration_ms", 0))
        input_shapes = node.get("input_shapes")
        if input_shapes is not None and input_shapes!= {}:
            stats[key]["input_shapes"].append(input_shapes)
        output_shapes = node.get("output_shapes")
        if output_shapes is not None and output_shapes!= {}:
            stats[key]["output_shapes"].append(output_shapes)
        meta = node.get("meta")
        if meta is not None and output_shapes != {}:
            stats[key]["meta"].append(meta)
        # stats[key]["nodes"].append(node)

        sub_nodes = node.get("sub_nodes", {})
        for sub_name, nodes in sub_nodes.items():
            for child in nodes:
                dfs(child, cur_path)

    dfs(root, ())
    return stats

from statistics import mean

def summarize_multi_rank_stats(rank_stats: list[dict]):
    merged = defaultdict(lambda: {
        "durations": [],
        "input_shapes": [],
        "output_shapes": [],
        "meta": [],
    })

    # merge across ranks
    for stats in rank_stats:
        for chain, item in stats.items():
            merged[chain]["durations"].extend(item.get("durations", []))
            merged[chain]["input_shapes"].extend(item.get("input_shapes", []))
            merged[chain]["output_shapes"].extend(item.get("output_shapes", []))
            merged[chain]["meta"].extend(item.get("meta", []))

    # summarize
    summary = {}

    for chain, item in merged.items():
        durations = item["durations"]

        summary[chain] = {
            "count": len(durations),
            "duration_avg_ms": mean(durations) if durations else None,

            # 不做平均，直接列出所有 rank / all calls 的观测
            "input_shapes": item["input_shapes"],
            "output_shapes": item["output_shapes"],
            "meta": item["meta"],
        }

    return summary

def expand_summary_with_stats(summary: dict) -> dict:
    root = {
        "name": "stat_root",
        "sub_nodes": {},
    }

    for chain, stats in summary.items():
        names = chain.split("->") if isinstance(chain, str) else list(chain)

        cur = root
        for name in names:
            sub_nodes = cur.setdefault("sub_nodes", {})

            if name not in sub_nodes:
                sub_nodes[name] = {
                    "name": name,
                    "sub_nodes": {},
                }

            cur = sub_nodes[name]

        # 关键：直接挂 stats，不动原有字段
        cur["stats"] = stats
    return root