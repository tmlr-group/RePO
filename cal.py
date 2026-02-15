import argparse
import csv
import os
from collections import Counter


def count_unique_strings_in_csv(filename, column_index=0, max_lines=100):
    """Count unique strings and frequencies in the first N rows of a CSV column."""
    string_counts = Counter()
    line_count = 0

    try:
        with open(filename, "r", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)

            for row in reader:
                if line_count >= max_lines:
                    break
                if row and len(row) > column_index:
                    string_value = row[column_index].strip()
                    if string_value:
                        string_counts[string_value] += 1
                line_count += 1

        num_unique = len(string_counts)
        print(f"Processed file '{filename}' first {line_count} rows.")
        return num_unique, string_counts
    except FileNotFoundError:
        print(f"Error: file not found '{filename}'")
        return 0, Counter()
    except Exception as e:
        print(f"Error while processing file: {e}")
        return 0, Counter()


def _read_first_row_metrics(csv_path):
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        first = next(reader, None)
        if first is None:
            raise ValueError(f"Empty file: {csv_path}")
        if "similarity" not in first or "success_rate" not in first:
            raise ValueError(f"Missing columns similarity/success_rate: {csv_path}")
        return float(first["similarity"]), float(first["success_rate"])


def calculate_weighted_success_rate(
    mopt_path,
    summary_files=("LogP_summary.csv", "MR_summary.csv", "QED_summary.csv"),
):
    """Read summary files and write *_wsr.csv where WSR=similarity*success_rate."""
    for filename in summary_files:
        src_path = os.path.join(mopt_path, filename)
        similarity, success_rate = _read_first_row_metrics(src_path)
        wsr = similarity * success_rate

        out_name = filename.replace("_summary.csv", "_wsr.csv")
        out_path = os.path.join(mopt_path, out_name)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["WSR"])
            writer.writerow([wsr])
        print(f"[wsr] {src_path} -> {out_path} (WSR={wsr:.6f})")


def _run_count(args):
    print(f"Analyzing file: {args.input_csv} (parsed as CSV)")
    print(f"Counting column {args.column_index + 1} in first {args.max_lines} rows...")
    count, counter_obj = count_unique_strings_in_csv(
        args.input_csv,
        column_index=args.column_index,
        max_lines=args.max_lines,
    )
    if count <= 0:
        print("No strings were counted.")
        return

    print(f"\nIn first {args.max_lines} rows, found {count} unique strings.")
    total_processed = sum(counter_obj.values())
    print(f"These unique strings appear {total_processed} times in total.")

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Count", "String"])
        for value, freq in counter_obj.most_common():
            writer.writerow([freq, value])
    print(f"Detailed results were saved to '{args.output_csv}'")


def _run_wsr(args):
    files = tuple([x.strip() for x in args.files.split(",") if x.strip()])
    calculate_weighted_success_rate(args.mopt_path, summary_files=files)


def build_parser():
    parser = argparse.ArgumentParser(description="Small CSV utilities (count / wsr)")
    parser.add_argument(
        "--mode",
        choices=["count", "wsr"],
        default="count",
        help="count: count unique strings; wsr: compute WSR",
    )

    # count mode args
    parser.add_argument(
        "--input_csv",
        type=str,
        default="predictions/qwen2.5-3b-instruct-sample-grpogrpo/open_generation/MolOpt/LogP.csv",
    )
    parser.add_argument("--column_index", type=int, default=0)
    parser.add_argument("--max_lines", type=int, default=100)
    parser.add_argument("--output_csv", type=str, default="unique_string_counts_top100.csv")

    # wsr mode args
    parser.add_argument(
        "--mopt_path",
        type=str,
        default="./predictions/xxx/open_generation/MolOpt/",
    )
    parser.add_argument(
        "--files",
        type=str,
        default="LogP_summary.csv,MR_summary.csv,QED_summary.csv",
        help="Comma-separated summary file names",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.mode == "count":
        _run_count(args)
    else:
        _run_wsr(args)

