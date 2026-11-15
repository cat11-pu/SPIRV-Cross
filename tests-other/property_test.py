#!/usr/bin/env python3
#
# Copyright 2026 The SPIRV-Cross Authors
# SPDX-License-Identifier: Apache-2.0 OR MIT

import argparse
import hashlib
import json
import os
import shutil
import signal
import struct
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


SPIRV_MAGIC = 0x07230203
EXPECTED_MALFORMED = 200
EXPECTED_VALID = 50
EXPECTED_CATEGORIES = {
    "illegal_type": 50,
    "empty_branch": 50,
    "deep_nesting": 50,
    "repeated_decoration": 50,
}

OPCODE_NAMES = {
    0: "OpNop",
    1: "OpUndef",
    5: "OpName",
    6: "OpMemberName",
    10: "OpExtension",
    11: "OpExtInstImport",
    12: "OpExtInst",
    14: "OpMemoryModel",
    15: "OpEntryPoint",
    16: "OpExecutionMode",
    17: "OpCapability",
    19: "OpTypeVoid",
    20: "OpTypeBool",
    21: "OpTypeInt",
    22: "OpTypeFloat",
    23: "OpTypeVector",
    24: "OpTypeMatrix",
    25: "OpTypeImage",
    26: "OpTypeSampler",
    27: "OpTypeSampledImage",
    28: "OpTypeArray",
    29: "OpTypeRuntimeArray",
    30: "OpTypeStruct",
    31: "OpTypeOpaque",
    32: "OpTypePointer",
    33: "OpTypeFunction",
    41: "OpConstantTrue",
    42: "OpConstantFalse",
    43: "OpConstant",
    44: "OpConstantComposite",
    45: "OpConstantSampler",
    46: "OpConstantNull",
    54: "OpFunction",
    55: "OpFunctionParameter",
    56: "OpFunctionEnd",
    57: "OpFunctionCall",
    59: "OpVariable",
    61: "OpLoad",
    62: "OpStore",
    65: "OpAccessChain",
    66: "OpInBoundsAccessChain",
    68: "OpArrayLength",
    71: "OpDecorate",
    72: "OpMemberDecorate",
    73: "OpDecorationGroup",
    74: "OpGroupDecorate",
    75: "OpGroupMemberDecorate",
    245: "OpPhi",
    246: "OpLoopMerge",
    247: "OpSelectionMerge",
    248: "OpLabel",
    249: "OpBranch",
    250: "OpBranchConditional",
    251: "OpSwitch",
    252: "OpKill",
    253: "OpReturn",
    254: "OpReturnValue",
    255: "OpUnreachable",
}

OOB_MARKERS = (
    "heap-buffer-overflow",
    "stack-buffer-overflow",
    "global-buffer-overflow",
    "dynamic-stack-buffer-overflow",
    "container-overflow",
    "use-after-free",
    "use-after-poison",
)

LEAK_MARKERS = (
    "LeakSanitizer",
    "detected memory leaks",
    "Direct leak of",
    "Indirect leak of",
)

UBSAN_MARKERS = (
    "runtime error:",
    "UndefinedBehaviorSanitizer",
)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_spirv(data):
    if len(data) < 20 or len(data) % 4:
        return {
            "valid_header": False,
            "error": "size is not a non-empty multiple of four bytes",
            "instructions": [],
        }

    words = struct.unpack("<%dI" % (len(data) // 4), data)
    if words[0] != SPIRV_MAGIC:
        return {
            "valid_header": False,
            "error": "bad SPIR-V magic 0x%08x" % words[0],
            "instructions": [],
        }

    instructions = []
    offset = 5
    while offset < len(words):
        first = words[offset]
        word_count = first >> 16
        opcode = first & 0xFFFF
        if word_count == 0:
            instructions.append(
                {
                    "word_offset": offset,
                    "byte_offset": offset * 4,
                    "word_count": 0,
                    "opcode": opcode,
                    "opcode_name": OPCODE_NAMES.get(opcode, "opcode_%d" % opcode),
                    "truncated": True,
                }
            )
            break

        instructions.append(
            {
                "word_offset": offset,
                "byte_offset": offset * 4,
                "word_count": word_count,
                "opcode": opcode,
                "opcode_name": OPCODE_NAMES.get(opcode, "opcode_%d" % opcode),
                "truncated": offset + word_count > len(words),
            }
        )
        offset += word_count

    return {
        "valid_header": True,
        "error": None,
        "instructions": instructions,
    }


def locate_offset(parsed, requested_words):
    instructions = parsed["instructions"]
    if not instructions:
        return {
            "requested_word_offset": requested_words,
            "requested_byte_offset": requested_words * 4,
            "exact": False,
            "instruction": None,
        }

    exact = next(
        (item for item in instructions if item["word_offset"] == requested_words),
        None,
    )
    if exact is not None:
        return {
            "requested_word_offset": requested_words,
            "requested_byte_offset": requested_words * 4,
            "exact": True,
            "instruction": exact,
        }

    preceding = None
    for item in instructions:
        if item["word_offset"] > requested_words:
            break
        preceding = item
    return {
        "requested_word_offset": requested_words,
        "requested_byte_offset": requested_words * 4,
        "exact": False,
        "instruction": preceding,
    }


def classify_result(returncode, stdout, stderr, timed_out):
    text = stderr.decode("utf-8", "replace")
    markers = []
    for marker in (
        "AddressSanitizer",
        "LeakSanitizer",
        "UndefinedBehaviorSanitizer",
        "runtime error:",
        "stack-overflow",
        *OOB_MARKERS,
        *LEAK_MARKERS,
        *UBSAN_MARKERS,
    ):
        if marker in text and marker not in markers:
            markers.append(marker)

    oob = any(marker in text for marker in OOB_MARKERS)
    leak = any(marker in text for marker in LEAK_MARKERS)
    ubsan = any(marker in text for marker in UBSAN_MARKERS)
    sanitizer = "AddressSanitizer" in text or leak or ubsan
    assertion = "Assertion `" in text or "assertion failed" in text.lower()
    fatal_signal = returncode < 0
    crash_codes = {133, 134, 135, 136, 137, 139}
    crash = (
        timed_out
        or fatal_signal
        or returncode in crash_codes
        or sanitizer
        or assertion
        or "terminate called" in text
    )
    normal_reject = not crash and returncode != 0
    accepted = not crash and returncode == 0
    failure = crash or oob or leak

    return {
        "returncode": returncode,
        "signal": -returncode if fatal_signal else None,
        "signal_name": signal.Signals(-returncode).name if fatal_signal else None,
        "timed_out": timed_out,
        "accepted": accepted,
        "normal_reject": normal_reject,
        "crash": crash,
        "out_of_bounds": oob,
        "leak": leak,
        "ubsan": ubsan,
        "failure": failure,
        "sanitizer_markers": markers,
        "stdout_bytes": len(stdout),
        "stdout_sha256": sha256_bytes(stdout),
    }


def run_process(binary, spv_path, extra_args, timeout, env):
    command = [str(binary), str(spv_path), *extra_args]
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            env=env,
            check=False,
        )
        timed_out = False
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = -signal.SIGKILL
        stdout = exc.stdout or b""
        stderr = exc.stderr or b""

    result = classify_result(returncode, stdout, stderr, timed_out)
    result.update(
        {
            "command": command,
            "stdout": stdout,
            "stderr": stderr,
        }
    )
    return result


def first_byte_difference(candidate, reference):
    limit = min(len(candidate), len(reference))
    for index in range(limit):
        if candidate[index] != reference[index]:
            break
    else:
        if len(candidate) == len(reference):
            return None
        index = limit

    prefix = candidate[:index]
    line = prefix.count(b"\n") + 1
    last_newline = prefix.rfind(b"\n")
    column = index + 1 if last_newline < 0 else index - last_newline
    start = max(0, index - 40)
    end = index + 80
    return {
        "byte_offset": index,
        "line": line,
        "column": column,
        "candidate_sha256": sha256_bytes(candidate),
        "reference_sha256": sha256_bytes(reference),
        "candidate_bytes": len(candidate),
        "reference_bytes": len(reference),
        "candidate_excerpt": candidate[start:end].decode("utf-8", "replace"),
        "reference_excerpt": reference[start:end].decode("utf-8", "replace"),
    }


def describe_failure(result):
    if result["timed_out"]:
        return "timeout"
    if result["out_of_bounds"]:
        return "out-of-bounds"
    if result["leak"]:
        return "leak"
    if result["ubsan"]:
        return "undefined-behavior"
    if "stack-overflow" in result["sanitizer_markers"]:
        return "stack-overflow"
    if result["signal_name"]:
        return result["signal_name"].lower()
    return "crash"


def first_stderr_line(stderr):
    for line in stderr.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if line:
            return line[:500]
    return ""


def build_candidate_environment(base_env):
    env = dict(base_env)
    env["ASAN_OPTIONS"] = (
        "detect_leaks=1:abort_on_error=1:halt_on_error=1:"
        "detect_stack_use_after_return=1:strict_string_checks=1"
    )
    env["LSAN_OPTIONS"] = "exitcode=23:report_objects=1"
    env["UBSAN_OPTIONS"] = "halt_on_error=1:print_stacktrace=1"
    return env


def validate_manifest(artifacts, manifest_name, expected_count, expected_categories):
    manifest_path = artifacts / manifest_name
    if not manifest_path.is_file():
        raise ValueError("missing manifest: %s" % manifest_path)

    entries = read_json(manifest_path)
    if not isinstance(entries, list):
        raise ValueError("%s must contain a JSON list" % manifest_path)
    if len(entries) != expected_count:
        raise ValueError(
            "%s has %d entries, expected %d"
            % (manifest_path, len(entries), expected_count)
        )

    ids = set()
    categories = Counter()
    for entry in entries:
        for field in ("id", "category", "file"):
            if field not in entry:
                raise ValueError("%s entry is missing %s" % (manifest_path, field))
        if entry["id"] in ids:
            raise ValueError("duplicate id %s in %s" % (entry["id"], manifest_path))
        ids.add(entry["id"])
        categories[entry["category"]] += 1

        spv_path = artifacts / entry["file"]
        if not spv_path.is_file():
            raise ValueError("missing corpus file: %s" % spv_path)
        parsed = parse_spirv(spv_path.read_bytes())
        if not parsed["valid_header"]:
            raise ValueError("%s is not a parseable SPIR-V container" % spv_path)

    if expected_categories is not None and dict(categories) != expected_categories:
        raise ValueError(
            "category counts are %s, expected %s"
            % (dict(categories), expected_categories)
        )
    return entries, categories


def save_counterexample(counterexamples_dir, entry, result, parsed, outcome):
    case_dir = counterexamples_dir / entry["id"]
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)

    source = Path(entry["_source"])
    shutil.copy2(source, case_dir / source.name)
    (case_dir / "stdout.txt").write_bytes(result["stdout"])
    (case_dir / "stderr.txt").write_bytes(result["stderr"])

    requested_words = int(entry.get("instruction_offset_words", 0))
    location = locate_offset(parsed, requested_words)
    instruction = location["instruction"]
    metadata = {
        "id": entry["id"],
        "category": entry["category"],
        "description": entry.get("description", ""),
        "instruction_offset_words": requested_words,
        "instruction_offset_bytes": requested_words * 4,
        "instruction_offset_exact": location["exact"],
        "instruction": instruction,
        "returncode": result["returncode"],
        "signal": result["signal"],
        "signal_name": result["signal_name"],
        "timed_out": result["timed_out"],
        "accepted": result["accepted"],
        "normal_reject": result["normal_reject"],
        "crash": result["crash"],
        "out_of_bounds": result["out_of_bounds"],
        "leak": result["leak"],
        "ubsan": result["ubsan"],
        "sanitizer_markers": result["sanitizer_markers"],
        "failure_kind": describe_failure(result),
        "stderr_first_line": first_stderr_line(result["stderr"]),
        "stdout_bytes": result["stdout_bytes"],
        "stdout_sha256": result["stdout_sha256"],
        "spv_sha256": sha256_file(source),
        "outcome": outcome,
        "command": result["command"],
        "stderr_excerpt": result["stderr"][:12000].decode("utf-8", "replace"),
    }
    (case_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return case_dir


def save_valid_counterexample(counterexamples_dir, entry, candidate, reference, diff):
    case_dir = counterexamples_dir / ("valid-" + entry["id"])
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)

    source = Path(entry["_source"])
    shutil.copy2(source, case_dir / source.name)
    (case_dir / "candidate.txt").write_bytes(candidate["stdout"])
    (case_dir / "reference.txt").write_bytes(reference["stdout"])
    (case_dir / "candidate.stderr.txt").write_bytes(candidate["stderr"])
    (case_dir / "reference.stderr.txt").write_bytes(reference["stderr"])
    metadata = {
        "id": entry["id"],
        "category": "valid_parity",
        "description": entry.get("description", ""),
        "candidate_returncode": candidate["returncode"],
        "reference_returncode": reference["returncode"],
        "first_difference": diff,
        "candidate_sanitizer_markers": candidate["sanitizer_markers"],
    }
    (case_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return case_dir


def markdown_summary(summary):
    malformed = summary["malformed"]
    valid = summary["valid"]
    lines = [
        "# SPIR-V Property Test Summary",
        "",
        "- Generated: `%s`" % summary["generated_at"],
        "- Source revision: `%s`" % summary["source_revision"],
        "- Candidate: `%s`" % summary["candidate"],
        "- Reference: `%s`" % summary["reference"],
        "- Target arguments: `%s`" % " ".join(summary["target_args"]),
        "",
        "## Malformed Corpus",
        "",
        "| Metric | Count |",
        "|---|---:|",
        "| Total | %d |" % malformed["total"],
        "| Clean accept | %d |" % malformed["accepted"],
        "| Clean reject | %d |" % malformed["rejected"],
        "| Crash | %d |" % malformed["crash"],
        "| Out of bounds | %d |" % malformed["out_of_bounds"],
        "| Leak | %d |" % malformed["leak"],
        "| UBSan | %d |" % malformed["ubsan"],
        "| Timeout | %d |" % malformed["timeout"],
        "| Failures | %d |" % malformed["failures"],
        "",
        "| Category | Total | Clean accept | Clean reject | Failures |",
        "|---|---:|---:|---:|---:|",
    ]
    for category in sorted(malformed["by_category"]):
        item = malformed["by_category"][category]
        lines.append(
            "| %s | %d | %d | %d | %d |"
            % (
                category,
                item["total"],
                item["accepted"],
                item["rejected"],
                item["failures"],
            )
        )

    lines.extend(
        [
            "",
            "## Valid Parity",
            "",
            "- Matched byte-for-byte: %d/%d" % (valid["matched"], valid["total"]),
            "- Candidate conversion errors: %d" % valid["candidate_errors"],
            "- Sanitizer failures: %d" % valid["sanitizer_failures"],
            "- Output mismatches: %d" % valid["mismatches"],
            "",
        ]
    )

    if malformed["failure_details"]:
        clusters = defaultdict(list)
        for item in malformed["failure_details"]:
            clusters[item["failure_kind"]].append(item["id"])
        lines.extend(
            [
                "## Failure Clusters",
                "",
                "| Kind | Count | IDs |",
                "|---|---:|---|",
            ]
        )
        for kind in sorted(clusters):
            lines.append(
                "| %s | %d | %s |"
                % (kind, len(clusters[kind]), ", ".join(clusters[kind]))
            )
        lines.append("")

        lines.extend(
            [
                "## Malformed Counterexamples",
                "",
                "| ID | Category | Offset (words/bytes) | Instruction | Kind |",
                "|---|---|---:|---|---|",
            ]
        )
        for item in malformed["failure_details"]:
            lines.append(
                "| %s | %s | %d / %d | %s | %s |"
                % (
                    item["id"],
                    item["category"],
                    item["instruction_offset_words"],
                    item["instruction_offset_bytes"],
                    item["instruction_name"],
                    item["failure_kind"],
                )
            )
        lines.append("")

    if valid["mismatch_details"]:
        lines.extend(
            [
                "## Valid Output Diffs",
                "",
                "| ID | Byte | Line:Column | Candidate bytes | Reference bytes |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for item in valid["mismatch_details"]:
            diff = item["first_difference"]
            if diff is None:
                lines.append(
                    "| %s | n/a | n/a | %d | %d |"
                    % (
                        item["id"],
                        item["candidate_bytes"],
                        item["reference_bytes"],
                    )
                )
            else:
                lines.append(
                    "| %s | %d | %d:%d | %d | %d |"
                    % (
                        item["id"],
                        diff["byte_offset"],
                        diff["line"],
                        diff["column"],
                        diff["candidate_bytes"],
                        diff["reference_bytes"],
                    )
                )
        lines.append("")

    lines.extend(
        [
            "## Verdict",
            "",
            "**%s**" % ("PASS" if summary["passed"] else "FAIL"),
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Property-test malformed and valid SPIR-V modules."
    )
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "property",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--target",
        choices=("glsl", "msl", "hlsl", "reflect"),
        default="glsl",
    )
    args = parser.parse_args()

    artifacts = args.artifacts.resolve()
    candidate = args.candidate.resolve()
    reference = args.reference.resolve()
    for binary in (candidate, reference):
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError("not an executable file: %s" % binary)

    target_args = [] if args.target == "glsl" else ["--" + args.target]
    malformed, malformed_categories = validate_manifest(
        artifacts,
        "manifest.malformed.json",
        EXPECTED_MALFORMED,
        EXPECTED_CATEGORIES,
    )
    valid, _ = validate_manifest(
        artifacts,
        "manifest.valid.json",
        EXPECTED_VALID,
        {"valid": EXPECTED_VALID},
    )

    for entry in malformed + valid:
        entry["_source"] = str(artifacts / entry["file"])

    candidate_env = build_candidate_environment(os.environ)
    reference_env = dict(os.environ)
    counterexamples_dir = artifacts / "counterexamples"
    counterexamples_dir.mkdir(parents=True, exist_ok=True)

    malformed_stats = {
        "total": len(malformed),
        "accepted": 0,
        "rejected": 0,
        "crash": 0,
        "out_of_bounds": 0,
        "leak": 0,
        "ubsan": 0,
        "timeout": 0,
        "failures": 0,
        "by_category": {},
        "failure_details": [],
    }
    category_stats = defaultdict(
        lambda: {
            "total": 0,
            "accepted": 0,
            "rejected": 0,
            "failures": 0,
        }
    )

    for entry in malformed:
        source = Path(entry["_source"])
        parsed = parse_spirv(source.read_bytes())
        result = run_process(
            candidate,
            source,
            target_args,
            args.timeout,
            candidate_env,
        )
        stats = category_stats[entry["category"]]
        stats["total"] += 1
        if result["failure"]:
            stats["failures"] += 1
            malformed_stats["failures"] += 1
        else:
            stats["accepted"] += int(result["accepted"])
            stats["rejected"] += int(result["normal_reject"])

        malformed_stats["accepted"] += int(result["accepted"])
        malformed_stats["rejected"] += int(result["normal_reject"])
        malformed_stats["crash"] += int(result["crash"])
        malformed_stats["out_of_bounds"] += int(result["out_of_bounds"])
        malformed_stats["leak"] += int(result["leak"])
        malformed_stats["ubsan"] += int(result["ubsan"])
        malformed_stats["timeout"] += int(result["timed_out"])

        if result["failure"]:
            location = locate_offset(
                parsed, int(entry.get("instruction_offset_words", 0))
            )
            instruction = location["instruction"]
            case_dir = save_counterexample(
                counterexamples_dir, entry, result, parsed, "failure"
            )
            malformed_stats["failure_details"].append(
                {
                    "id": entry["id"],
                    "category": entry["category"],
                    "description": entry.get("description", ""),
                    "instruction_offset_words": location["requested_word_offset"],
                    "instruction_offset_bytes": location["requested_byte_offset"],
                    "instruction_name": (
                        instruction["opcode_name"] if instruction else "unknown"
                    ),
                    "returncode": result["returncode"],
                    "signal": result["signal_name"],
                    "crash": result["crash"],
                    "out_of_bounds": result["out_of_bounds"],
                    "leak": result["leak"],
                    "ubsan": result["ubsan"],
                    "timed_out": result["timed_out"],
                    "sanitizer_markers": result["sanitizer_markers"],
                    "failure_kind": describe_failure(result),
                    "stderr_first_line": first_stderr_line(result["stderr"]),
                    "counterexample_dir": str(case_dir),
                }
            )

    malformed_stats["by_category"] = dict(sorted(category_stats.items()))

    valid_stats = {
        "total": len(valid),
        "matched": 0,
        "candidate_errors": 0,
        "sanitizer_failures": 0,
        "mismatches": 0,
        "mismatch_details": [],
    }

    for entry in valid:
        source = Path(entry["_source"])
        candidate_result = run_process(
            candidate,
            source,
            target_args,
            args.timeout,
            candidate_env,
        )
        reference_result = run_process(
            reference,
            source,
            target_args,
            args.timeout,
            reference_env,
        )

        diff = first_byte_difference(
            candidate_result["stdout"], reference_result["stdout"]
        )
        candidate_error = (
            candidate_result["returncode"] != 0
            or reference_result["returncode"] != 0
            or diff is not None
            or candidate_result["stderr"] != reference_result["stderr"]
        )
        sanitizer_failure = (
            candidate_result["crash"]
            or candidate_result["out_of_bounds"]
            or candidate_result["leak"]
            or candidate_result["ubsan"]
        )

        valid_stats["candidate_errors"] += int(candidate_result["returncode"] != 0)
        valid_stats["sanitizer_failures"] += int(sanitizer_failure)

        if candidate_error or sanitizer_failure:
            if diff is not None:
                valid_stats["mismatches"] += 1
            case_dir = save_valid_counterexample(
                counterexamples_dir,
                entry,
                candidate_result,
                reference_result,
                diff,
            )
            valid_stats["mismatch_details"].append(
                {
                    "id": entry["id"],
                    "description": entry.get("description", ""),
                    "candidate_returncode": candidate_result["returncode"],
                    "reference_returncode": reference_result["returncode"],
                    "candidate_bytes": len(candidate_result["stdout"]),
                    "reference_bytes": len(reference_result["stdout"]),
                    "first_difference": diff,
                    "sanitizer_markers": candidate_result["sanitizer_markers"],
                    "counterexample_dir": str(case_dir),
                }
            )
        else:
            valid_stats["matched"] += 1

    source_revision = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    ).stdout.strip()

    passed = (
        malformed_stats["failures"] == 0
        and valid_stats["matched"] == valid_stats["total"]
    )
    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_revision": source_revision or "unknown",
        "candidate": str(candidate),
        "reference": str(reference),
        "target": args.target,
        "target_args": target_args,
        "timeout_seconds": args.timeout,
        "corpus": {
            "malformed": len(malformed),
            "valid": len(valid),
            "malformed_categories": malformed_categories,
        },
        "malformed": malformed_stats,
        "valid": valid_stats,
        "passed": passed,
    }

    summary_json = artifacts / "summary.json"
    summary_md = artifacts / "SUMMARY.md"
    summary_json.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    summary_md.write_text(markdown_summary(summary) + "\n", encoding="utf-8")

    print(
        "malformed=%d failures=%d crashes=%d oob=%d leaks=%d timeouts=%d"
        % (
            malformed_stats["total"],
            malformed_stats["failures"],
            malformed_stats["crash"],
            malformed_stats["out_of_bounds"],
            malformed_stats["leak"],
            malformed_stats["timeout"],
        )
    )
    print(
        "valid=%d matched=%d mismatches=%d candidate_errors=%d sanitizer_failures=%d"
        % (
            valid_stats["total"],
            valid_stats["matched"],
            valid_stats["mismatches"],
            valid_stats["candidate_errors"],
            valid_stats["sanitizer_failures"],
        )
    )
    print("summary=%s" % summary_md)
    print("json=%s" % summary_json)
    for item in malformed_stats["failure_details"]:
        print(
            "counterexample id=%s category=%s offset=%dw/%dB instruction=%s"
            % (
                item["id"],
                item["category"],
                item["instruction_offset_words"],
                item["instruction_offset_bytes"],
                item["instruction_name"],
            )
        )

    return 0 if passed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print("property_test.py: %s" % exc, file=sys.stderr)
        sys.exit(2)
