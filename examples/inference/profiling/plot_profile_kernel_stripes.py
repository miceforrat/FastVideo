import argparse
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


KERNELS = [
    "prepare_ms",
    "cross_attn_core_ms",
    "hs_norm_ms",
    "to_q_ms",
    "to_k_ms",
    "to_v_ms",
    "q_rms_norm_ms",
    "core_attn_ms",
    "to_out_ms",
    "fc_in_ms",
    "act_ms",
    "fc_out_ms",
]


COLORS = {
    "prepare_ms": (160, 160, 160),
    "cross_attn_core_ms": (255, 180, 80),
    "hs_norm_ms": (120, 180, 255),
    "to_q_ms": (70, 130, 255),
    "to_k_ms": (50, 100, 220),
    "to_v_ms": (30, 80, 190),
    "q_rms_norm_ms": (100, 220, 220),
    "core_attn_ms": (220, 60, 60),
    "to_out_ms": (255, 120, 120),
    "fc_in_ms": (120, 220, 120),
    "act_ms": (80, 180, 80),
    "fc_out_ms": (40, 140, 40),
}


def load_font(size):
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def infer_model(row):
    num_heads = int(row["num_heads"])
    head_dim = int(row["head_dim"])
    ffn_dim = int(row["ffn_dim"])

    if num_heads == 12 and head_dim == 128 and ffn_dim == 8960:
        return "t2v-1.3B"
    if num_heads == 40 and head_dim == 128 and ffn_dim == 13824:
        return "t2v-14B"
    return f"h{head_dim}x{num_heads}_ffn{ffn_dim}"


def safe_name(s):
    return str(s).replace("/", "_").replace(" ", "_")


def x_of_index(i, col_width, gap):
    return i * (col_width + gap)


def draw_centered_text(draw, box, text, font, fill):
    x0, y0, x1, y1 = box
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = x0 + max(0, (x1 - x0 - tw) // 2)
    y = y0 + max(0, (y1 - y0 - th) // 2)
    draw.text((x, y), text, font=font, fill=fill)


def draw_group_axis(draw, df, group_cols, y0, y1, col_width, gap, line_color, text_color, prefix, font):
    for key, g in df.groupby(group_cols, sort=False):
        value = key[-1] if isinstance(key, tuple) else key

        start = int(g.index.min())
        end = int(g.index.max())

        x0 = x_of_index(start, col_width, gap)
        x1 = x_of_index(end, col_width, gap) + col_width - 1

        draw.line([x0, y0, x0, y1], fill=line_color, width=2)
        draw.line([x1, y0, x1, y1], fill=line_color, width=1)
        draw.line([x0, y0, x1, y0], fill=line_color, width=1)

        label = f"{prefix}{int(value)}"
        if x1 - x0 >= 60:
            draw_centered_text(draw, (x0, y0, x1, y1), label, font, text_color)


def draw_one_model(df, kernels, model, args, font_axis):
    df = df.copy().reset_index(drop=True)

    n = len(df)
    stripe_h = args.height
    axis_h = args.axis_height
    title_h = args.title_height

    width = n * args.col_width + max(0, n - 1) * args.gap
    total_h = title_h + stripe_h + axis_h

    img = Image.new("RGB", (width, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    title_font = load_font(args.title_font_size)
    draw_centered_text(
        draw,
        (0, 0, width - 1, title_h - 1),
        f"{model} kernel time percentage stripes",
        title_font,
        (0, 0, 0),
    )

    y_offset = title_h

    for i, row in df.iterrows():
        x0 = x_of_index(i, args.col_width, args.gap)
        x1 = x0 + args.col_width - 1

        total = float(row["kernel_total_ms"])
        vals = [(k, float(row[k]) / total) for k in kernels]
        vals.sort(key=lambda x: x[1], reverse=True)

        y = y_offset
        used = 0

        for j, (k, pct) in enumerate(vals):
            if j == len(vals) - 1:
                seg_h = stripe_h - used
            else:
                seg_h = int(round(pct * stripe_h))

            if seg_h <= 0:
                continue

            y0 = y
            y1 = min(y_offset + stripe_h - 1, y + seg_h - 1)

            draw.rectangle([x0, y0, x1, y1], fill=COLORS.get(k, (0, 0, 0)))

            y += seg_h
            used += seg_h

            if y >= y_offset + stripe_h:
                break

    axis_y0 = title_h + stripe_h
    draw.rectangle([0, axis_y0, width, total_h], fill=(255, 255, 255))

    seqlen_y0 = axis_y0
    text_y0 = axis_y0 + axis_h // 2

    draw_group_axis(
        draw, df,
        ["seqlen"],
        seqlen_y0, text_y0,
        args.col_width, args.gap,
        (80, 80, 80), (0, 0, 0),
        "SeqLen=", font_axis,
    )

    draw_group_axis(
        draw, df,
        ["seqlen", "text_len"],
        text_y0, axis_y0 + axis_h,
        args.col_width, args.gap,
        (150, 150, 150), (60, 60, 60),
        "PromptLen=", font_axis,
    )

    out = args.out_pattern.format(model=safe_name(model))
    img.save(out)

    segments = []
    for keys, g in df.groupby(["seqlen", "text_len"], sort=False):
        seqlen, text_len = keys
        segments.append({
            "model": model,
            "seqlen": int(seqlen),
            "text_len": int(text_len),
            "start_col": int(g.index.min()),
            "end_col": int(g.index.max()),
            "num_configs": int(len(g)),
        })

    seg_df = pd.DataFrame(segments)
    seg_out = args.segment_pattern.format(model=safe_name(model))
    seg_df.to_csv(seg_out, index=False)

    print(f"Saved image: {out}")
    print(f"Saved segments: {seg_out}")
    print(f"{model}: valid configs={n}, image={width}x{total_h}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="logs/block_profile_chunk_idx/profile_results.csv")
    parser.add_argument("--out-pattern", default="plots/profile_kernel_stripes_{model}.png")
    parser.add_argument("--segment-pattern", default="plots/profile_group_segments_{model}.csv")
    parser.add_argument("--legend-out", default="plots/profile_kernel_legend.png")
    parser.add_argument("--height", type=int, default=2000)
    parser.add_argument("--col-width", type=int, default=2)
    parser.add_argument("--gap", type=int, default=0)
    parser.add_argument("--axis-height", type=int, default=220)
    parser.add_argument("--title-height", type=int, default=60)
    parser.add_argument("--axis-font-size", type=int, default=28)
    parser.add_argument("--title-font-size", type=int, default=32)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df.columns = df.columns.str.strip()

    kernels = [k for k in KERNELS if k in df.columns]

    numeric_cols = [
        "head_dim", "num_heads", "ffn_dim",
        "seqlen", "text_len",
        "batch_size", "chunk_size", "chunk_idx",
        "kv_cache_frames", "kv_cache_tokens",
        "fwd_block_ms",
    ] + kernels

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    required_cols = ["num_heads", "head_dim", "ffn_dim", "seqlen", "text_len"] + kernels
    df = df.dropna(subset=[c for c in required_cols if c in df.columns])

    df["kernel_total_ms"] = df[kernels].sum(axis=1)
    df = df[df["kernel_total_ms"] > 0].copy()

    df["model"] = df.apply(infer_model, axis=1)

    int_cols = [
        "head_dim", "num_heads", "ffn_dim",
        "seqlen", "text_len",
        "batch_size", "chunk_size", "chunk_idx",
        "kv_cache_frames", "kv_cache_tokens",
    ]
    for c in int_cols:
        if c in df.columns:
            df[c] = df[c].astype(int)

    model_order = {"t2v-1.3B": 0, "t2v-14B": 1}
    df["model_order"] = df["model"].map(model_order).fillna(99).astype(int)

    sort_cols = [
        "model_order",
        "seqlen",
        "text_len",
        "kv_cache_frames",
        "batch_size",
        "chunk_size",
        "chunk_idx",
    ]
    sort_cols = [c for c in sort_cols if c in df.columns]
    df = df.sort_values(by=sort_cols, ascending=True).reset_index(drop=True)

    font_axis = load_font(args.axis_font_size)

    for model in df["model"].drop_duplicates().tolist():
        sub = df[df["model"] == model].copy().reset_index(drop=True)
        draw_one_model(sub, kernels, model, args, font_axis)

    # kernel 图例，保持原 kernel 顺序
    legend_font = load_font(24)
    legend_h = 36 * len(kernels) + 16
    legend_w = 520
    legend = Image.new("RGB", (legend_w, legend_h), (255, 255, 255))
    d = ImageDraw.Draw(legend)

    for i, k in enumerate(kernels):
        y = i * 36 + 8
        d.rectangle([12, y, 42, y + 24], fill=COLORS.get(k, (0, 0, 0)))
        label = k.replace("_ms", "")
        d.text((55, y - 1), label, font=legend_font, fill=(0, 0, 0))

    legend.save(args.legend_out)
    print(f"Saved legend: {args.legend_out}")


if __name__ == "__main__":
    main()