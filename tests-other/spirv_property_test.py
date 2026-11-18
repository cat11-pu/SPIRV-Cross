#!/usr/bin/env python3
"""
Run the malformed and valid SPIR-V property suites.

Each malformed module is executed in its own process so that sanitizer findings
and crashes are attributable to a single input. The valid suite translates each
module with the ordinary GLSL frontend and compares the result byte-for-byte
with the checked-in reference.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

KINDS = ("truncation", "illegal_type", "decoration_oob", "duplicate_entry")


def repo_root():
    return Path(__file__).resolve().parent.parent


def run_process(argv, timeout, env):
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            env=env,
            check=False,
        )
        return proc.returncode, proc.stdout, proc.stderr, (time.perf_counter() - start) * 1000.0
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or b""
        stderr = exc.stderr or b""
        return None, stdout, stderr, (time.perf_counter() - start) * 1000.0


def decode(data):
    return data.decode("utf-8", "replace")


def classify(returncode, stderr):
    if "LeakSanitizer" in stderr or "detected memory leaks" in stderr:
        return "leak_sanitizer"
    if "AddressSanitizer" in stderr:
        return "address_sanitizer"
    if "UndefinedBehaviorSanitizer" in stderr or "runtime error:" in stderr:
        return "undefined_behavior"
    if returncode is None:
        return "timeout"
    if returncode < 0 or returncode in (6, 87, 88, 134, 139):
        return "crash"
    if returncode != 0:
        return "unexpected_exit"
    return "safe"


def sanitizer_diagnostic(stderr):
    for line in stderr.splitlines():
        if (
            "AddressSanitizer" in line
            or "LeakSanitizer" in line
            or "runtime error:" in line
            or "SUMMARY:" in line
        ):
            return line.strip()
    return ""


def source_for_valid(root, spv):
    parts = spv.stem.split("-", 1)
    if len(parts) != 2:
        raise RuntimeError("Unexpected valid module name: {}".format(spv.name))
    stage, name = parts
    candidates = sorted((root / "shaders" / stage).glob(name + ".*"))
    candidates = [path for path in candidates if path.is_file()]
    if len(candidates) != 1:
        raise RuntimeError("Cannot resolve source for {}: {}".format(spv, candidates))
    source = candidates[0]
    reference = root / "reference" / "shaders" / source.relative_to(root / "shaders")
    return source, reference


def first_mismatch(expected, actual):
    limit = min(len(expected), len(actual))
    for offset in range(limit):
        if expected[offset] != actual[offset]:
            return offset
    if len(expected) != len(actual):
        return limit
    return None


def validate_manifest(manifest):
    errors = []
    if len(manifest) != 200:
        errors.append("expected 200 malformed cases, found {}".format(len(manifest)))
    counts = Counter(case.get("kind") for case in manifest)
    for kind in KINDS:
        if counts[kind] != 50:
            errors.append("expected 50 {} cases, found {}".format(kind, counts[kind]))
    for index, case in enumerate(manifest):
        expected = KINDS[index % len(KINDS)]
        if case.get("kind") != expected:
            errors.append(
                "case {} is {}, expected alternating {}".format(
                    case.get("case"), case.get("kind"), expected
                )
            )
            break
    return errors


def main():
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--artifacts", type=Path, default=root / "artifacts" / "spirv-fuzz")
    parser.add_argument(
        "--runner", type=Path, default=root / ".build-asan" / "spirv-cross-fuzz-runner"
    )
    parser.add_argument("--cross", type=Path, default=root / ".build-asan" / "spirv-cross")
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()

    artifacts = args.artifacts.resolve()
    malformed_dir = artifacts / "malformed"
    valid_dir = artifacts / "valid"
    translations_dir = artifacts / "translations"
    counterexamples_dir = artifacts / "counterexamples"
    manifest_path = malformed_dir / "manifest.json"

    required = [args.runner, args.cross, manifest_path, valid_dir]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("Missing required test inputs or binaries:", file=sys.stderr)
        for path in missing:
            print("  " + path, file=sys.stderr)
        return 2

    manifest = json.loads(manifest_path.read_text())
    manifest_errors = validate_manifest(manifest)
    valid_modules = sorted(valid_dir.glob("*.spv"))
    if len(valid_modules) != 50:
        manifest_errors.append("expected 50 valid modules, found {}".format(len(valid_modules)))
    if manifest_errors:
        print("Suite structure is invalid:", file=sys.stderr)
        for error in manifest_errors:
            print("  " + error, file=sys.stderr)
        return 2

    env = os.environ.copy()
    env["ASAN_OPTIONS"] = "detect_leaks=1:abort_on_error=1:allocator_may_return_null=1"
    env["LSAN_OPTIONS"] = "exitcode=23:report_objects=1"
    env["UBSAN_OPTIONS"] = "halt_on_error=1:print_stacktrace=1:exitcode=88"

    shutil.rmtree(counterexamples_dir, ignore_errors=True)
    counterexamples_dir.mkdir(parents=True)
    translations_dir.mkdir(parents=True, exist_ok=True)

    results = []
    counters = Counter()
    by_kind = {
        kind: {
            "total": 0,
            "crashes": 0,
            "address_sanitizer": 0,
            "leak_sanitizer": 0,
            "undefined_behavior": 0,
            "timeouts": 0,
            "unexpected_exits": 0,
        }
        for kind in KINDS
    }

    print("Running {} malformed modules...".format(len(manifest)))
    for index, case in enumerate(manifest):
        path = Path(case["path"])
        if not path.is_absolute():
            path = malformed_dir / path.name
        returncode, stdout, stderr, elapsed_ms = run_process(
            [str(args.runner.resolve()), str(path)], args.timeout, env
        )
        stdout_text = decode(stdout)
        stderr_text = decode(stderr)
        classification = classify(returncode, stderr_text)
        counters[classification] += 1

        kind_stats = by_kind[case["kind"]]
        kind_stats["total"] += 1
        if classification == "crash":
            kind_stats["crashes"] += 1
        elif classification == "address_sanitizer":
            kind_stats["address_sanitizer"] += 1
        elif classification == "leak_sanitizer":
            kind_stats["leak_sanitizer"] += 1
        elif classification == "undefined_behavior":
            kind_stats["undefined_behavior"] += 1
        elif classification == "timeout":
            kind_stats["timeouts"] += 1
        elif classification == "unexpected_exit":
            kind_stats["unexpected_exits"] += 1

        result = dict(case)
        result.update(
            {
                "path": str(path),
                "classification": classification,
                "returncode": returncode,
                "elapsed_ms": round(elapsed_ms, 3),
                "stdout": stdout_text,
                "stderr": stderr_text,
                "diagnostic": sanitizer_diagnostic(stderr_text),
            }
        )
        results.append(result)

        if classification != "safe":
            target = counterexamples_dir / path.name
            shutil.copy2(path, target)
            metadata = {key: value for key, value in result.items() if key != "stderr"}
            metadata["stderr"] = stderr_text[:16384]
            metadata["counterexample_file"] = str(target)
            (counterexamples_dir / (path.stem + ".json")).write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n"
            )
            print(
                "  FAIL {} {} offset={}:: {}".format(
                    case["case"],
                    case["kind"],
                    case["offset"],
                    classification,
                )
            )
        elif (index + 1) % 25 == 0:
            print("  checked {}/{}".format(index + 1, len(manifest)))

    controlled_rejections = sum(
        1
        for result in results
        if result["classification"] == "safe"
        and "PASS backend=" not in result["stdout"]
        and "REJECT backend=" in result["stdout"]
    )
    accepted = sum(
        1
        for result in results
        if result["classification"] == "safe" and "PASS backend=" in result["stdout"]
    )
    failure_count = len(results) - sum(
        1 for result in results if result["classification"] == "safe"
    )

    valid_results = []
    print("Comparing {} valid translations...".format(len(valid_modules)))
    for index, spv in enumerate(valid_modules):
        source, reference = source_for_valid(root, spv)
        returncode, stdout, stderr, elapsed_ms = run_process(
            [str(args.cross.resolve()), str(spv)], args.timeout, env
        )
        expected = reference.read_bytes() if reference.exists() else b""
        mismatch_offset = first_mismatch(expected, stdout)
        byte_equal = returncode == 0 and mismatch_offset is None and not stderr
        translation = translations_dir / (spv.stem + ".glsl")
        translation.write_bytes(stdout)
        result = {
            "source": str(source.relative_to(root)) if source.is_relative_to(root) else str(source),
            "stage": source.parent.name,
            "spv": str(spv),
            "reference": str(reference),
            "translation": str(translation),
            "returncode": returncode,
            "stderr": decode(stderr),
            "elapsed_ms": round(elapsed_ms, 3),
            "byte_equal": byte_equal,
            "mismatch_offset": mismatch_offset,
            "expected_size": len(expected),
            "actual_size": len(stdout),
        }
        valid_results.append(result)
        if not byte_equal:
            print(
                "  FAIL {} mismatch_offset={} rc={}".format(
                    spv.name, mismatch_offset, returncode
                )
            )
        elif (index + 1) % 10 == 0:
            print("  compared {}/{}".format(index + 1, len(valid_modules)))

    mismatch_details = [result for result in valid_results if not result["byte_equal"]]
    baseline_counterexamples = sorted(
        (artifacts / "baseline-counterexamples").glob("*.spv")
    )
    report = {
        "commit": subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            text=True,
        ).stdout.strip(),
        "sanitizers": {
            "address": True,
            "undefined": True,
            "leak": True,
            "runner": str(args.runner.resolve()),
            "cross": str(args.cross.resolve()),
        },
        "malformed": {
            "total": len(results),
            "controlled_rejections": controlled_rejections,
            "accepted": accepted,
            "crashes": counters["crash"],
            "address_sanitizer": counters["address_sanitizer"],
            "leak_sanitizer": counters["leak_sanitizer"],
            "undefined_behavior": counters["undefined_behavior"],
            "timeouts": counters["timeout"],
            "unexpected_exits": counters["unexpected_exit"],
            "counterexamples": failure_count,
            "by_kind": by_kind,
            "results": results,
        },
        "valid": {
            "total": len(valid_results),
            "byte_equal": len(valid_results) - len(mismatch_details),
            "mismatches": len(mismatch_details),
            "mismatch_details": mismatch_details,
            "results": valid_results,
        },
        "baseline_counterexamples": len(baseline_counterexamples),
    }

    (artifacts / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    summary = [
        "# SPIR-V malformed-input property test",
        "",
        "- malformed total: {}".format(report["malformed"]["total"]),
        "- controlled rejections: {}".format(controlled_rejections),
        "- accepted by at least one backend: {}".format(accepted),
        "- crashes: {}".format(report["malformed"]["crashes"]),
        "- address sanitizer findings: {}".format(report["malformed"]["address_sanitizer"]),
        "- leak sanitizer findings: {}".format(report["malformed"]["leak_sanitizer"]),
        "- undefined behavior findings: {}".format(report["malformed"]["undefined_behavior"]),
        "- timeouts: {}".format(report["malformed"]["timeouts"]),
        "- unexpected exits: {}".format(report["malformed"]["unexpected_exits"]),
        "- counterexamples: {}".format(failure_count),
        "- valid modules: {}".format(report["valid"]["total"]),
        "- byte-identical translations: {}".format(report["valid"]["byte_equal"]),
        "- reference mismatches: {}".format(report["valid"]["mismatches"]),
        "- pre-fix baseline counterexamples retained: {}".format(
            len(baseline_counterexamples)
        ),
        "",
        "## Malformed cases by kind",
        "",
        "| kind | cases | crashes | ASan | LSan | UBSan | timeouts |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for kind in KINDS:
        stats = by_kind[kind]
        summary.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                kind,
                stats["total"],
                stats["crashes"],
                stats["address_sanitizer"],
                stats["leak_sanitizer"],
                stats["undefined_behavior"],
                stats["timeouts"],
            )
        )

    summary.extend(["", "## Counterexamples", ""])
    failures = [result for result in results if result["classification"] != "safe"]
    if failures:
        for result in failures:
            summary.append(
                "- `{}` `{}` at byte offset {} (instruction offset {}): {} - {}".format(
                    result["case"],
                    result["kind"],
                    result["offset"],
                    result["instruction_offset"],
                    result["classification"],
                    result["diagnostic"] or result["stderr"].splitlines()[0],
                )
            )
    else:
        summary.append(
            "No counterexamples: all {} malformed modules were handled safely.".format(
                len(results)
            )
        )
        if baseline_counterexamples:
            summary.append("")
            summary.append(
                "The {} counterexamples from the pre-fix baseline are retained under "
                "`baseline-counterexamples/`.".format(len(baseline_counterexamples))
            )

    if mismatch_details:
        summary.extend(["", "## Valid translation mismatches", ""])
        for result in mismatch_details:
            summary.append(
                "- `{}` first difference at byte offset {} (rc={}).".format(
                    Path(result["spv"]).name,
                    result["mismatch_offset"],
                    result["returncode"],
                )
            )

    (artifacts / "summary.md").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))

    if manifest_errors:
        return 2
    if failure_count or mismatch_details:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
