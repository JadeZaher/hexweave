"""Render handwritten QA against the pinned T1 documentation; see README.md."""

import hashlib
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
DRIVE = HERE.parents[3] / "f-text-drive-20261024"
PIN = HERE / "source-pins.json"
SOURCES = {
    "T01": ("train", "docs/TRAINING-PULSE.md"),
    "T02": ("train", "docs/TEXT-PULSE.md"),
    "T03": ("train", "src/text/AGENTS.md"),
    "T04": ("train", "src/ternary/AGENTS.md"),
    "D01": ("development", "docs/BENCHMARKS.md"),
    "D02": ("development", "bench/AGENTS.md"),
    "D03": ("development", "benchmarks/fixtures/README.md"),
    "E01": ("reserved_evaluation_source", "docs/ARCHITECTURE.md"),
    "E02": ("reserved_evaluation_source", "src/model/AGENTS.md"),
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def excerpt(source, paragraph, start=None, stop=None):
    text = TEXT[source]
    part = re.split(r"\n\s*\n", text)[paragraph]
    if start is not None:
        part = part[part.index(start):]
    if stop is not None:
        part = part[:part.index(stop)]
    part = part.rstrip()
    if text.count(part) != 1:
        raise ValueError("excerpt must identify one exact source span")
    offset = text.index(part)
    begin = len(text[:offset].encode("utf-8"))
    end = begin + len(part.encode("utf-8"))
    return part, begin, end


# Handwritten questions and concise reference answers, not generated variants.
TRAIN = [
    ("T01", 1, "pulse_entry_points", "Which functions create the training pulse and combine its task and routing credit?", "TrainingPulse::forward creates the pulse; Gradients::add combines task and routing credit."),
    ("T01", 4, "pulse_objective", "Does the documented tile pulse use an adversarial generator/discriminator objective?", "No. It shares a latent representation across three control heads and does not use a GAN objective."),
    ("T01", 8, "pulse_shift_semantics", "What does shift change, and which quantities remain floating point?", "Shift modulates curve output. Latents, scales, activations and surrogate derivatives remain floating point."),
    ("T01", 7, "pulse_gain_equation", "What equation defines gain from the gain head?", "gain = 1 + 0.125 tanh(a_gain)"),
    ("T01", 10, "pulse_indirect_route_coupling", "Can a local predicted update change future routing without direct local route-score credit?", "Yes. A local update can change shared U and hence future routing; direct route-score credit is anchor-only."),
    ("T01", 11, "pulse_conditional_derivative", "Which inputs are detached in the returned conditional-path derivative?", "Routing choices, routing-feature input paths and history are detached."),
    ("T01", 14, "pulse_ablation_interpretation", "Why does the anchor-only comparison fail to isolate the shared trunk's benefit?", "It changes local updates and history while retaining U/V in both arms; a matched coupling ablation is needed."),
    ("T01", 15, "text_v3_artifact_identity", "Which protocol, tile schema and inspection snapshot version are documented here?", "The protocol is text-pulse-v3, tile schema is 2 and inspection snapshot is v3. Snapshots cannot resume."),
    ("T02", 3, "text_route_geometry", "How many tiles are used, and can routes revisit them?", "Seven hex tiles are used. Learned neighbor/halt routes can revisit tiles, with a three-visit cap."),
    ("T02", 8, "text_fixture_dimensions", "What are this text fixture's width, vocabulary and context, and does it use attention?", "Width 16, a 256-byte vocabulary and an eight-byte weighted embedding context; it has no attention."),
    ("T02", 10, "text_local_target_exclusion", "Do local updates inspect later tile weights or the current target?", "No. Local updates inspect neither; anchors inspect the realized path and obtain real task corrections."),
    ("T02", 13, "text_owner_launch_order", "How does the Windows supervisor establish ownership before numerical execution?", "It starts a suspended, below-normal child and assigns a Job Object before execution; ownership failure closes admission."),
    ("T02", 13, "text_owner_memory_scope", "What child committed-memory cap is intended, and is the Python owner inside that Job?", "The intended child cap is 512 MiB; actual enforced limits are in the terminal receipt. The Python owner is outside the Job."),
    ("T02", 14, "text_run_bounds", "What maximum opportunity count and deadline are documented, and are retries enabled?", "At most 1,024 opportunities and a 120-second deadline; there is no retry."),
    ("T02", 14, "text_log_guard", "Is the log limit a hard operating-system write quota?", "No. Logs have a polled 1 MiB detection threshold, not an OS write quota."),
    ("T02", 14, "text_artifact_retention", "Where are text-run artifacts stored, and are they automatically removed?", "Under ignored work/text/. They are retained for diagnosis, not automatically removed."),
    ("T02", 16, "text_exposure_units", "Which exposure count must be separated from predicted updates and step opportunities?", "Count actual anchor target tokens separately, and compare modes at equal actual target exposure."),
    ("T03", 3, "ternary_parameter_update", "Are float master weights maintained, and how far can one threshold-one update move a trit?", "There are no float master parameters. Threshold-one signed evidence moves a trit one adjacent step."),
    ("T03", 6, "text_activation_curve", "What fixed knot coordinates and endpoint rule define the text activation curve?", "Abscissae are [-2,0,2]; ordinates are [-1,0,1]+ternary_offsets/4. Endpoints extrapolate linearly."),
    ("T03", 6, "text_route_score_credit", "What score credit do hard sampled routing choices receive at anchors?", "(task_loss - previous_training_baseline) * grad(log probability), with routing features detached."),
    ("T03", 9, "pulse_matrix_shapes", "What are the shapes of U and V for width d and legal route count r?", "U is d by 2d; V is (2d + r) by d."),
    ("T03", 9, "pulse_modulation_ranges", "What ranges are given for gain and shift?", "Gain lies in (0.875,1.125); shift lies in (-0.125,0.125)."),
    ("T03", 10, "pulse_route_logit_clipping", "Are the route logits clipped by a tanh head?", "No. Route logits are finite but are not clipped by a tanh head."),
    ("T03", 11, "text_node_version_commit", "Which components does Node::apply commit under one node version?", "The transform, curve, router, pulse context and pulse heads."),
    ("T03", 12, "text_policy_gradient_boundary", "Does policy credit return into earlier tiles or through y into local transform parameters?", "No. Policy gradients do not return into earlier tiles or through y into local transform/curve parameters."),
    ("T03", 13, "text_local_parameter_credit", "Which pulse parameters receive predicted task credit on local steps?", "U and the gain/shift rows of V receive it; local steps do not directly update route-head rows."),
    ("T03", 15, "text_t1_detachment", "What does route-to-U-detached remove, and which route updates are preserved?", "It removes route-derived correction to U. All V rows and R updates remain, as do returned task corrections.", None, "See ["),
    ("T03", 16, "text_predictor_error_denominator", "How is the predictor-error diagnostic averaged?", "Accumulated per-update MSE is divided by predictor_updates, including replay."),
    ("T03", 17, "text_predictor_inputs", "What does the separate shared correction predictor map to a predicted correction?", "Input, pulse-conditioned output, history and position; it is separate from the per-tile conditioning pulse."),
    ("T03", 19, "text_local_tape_lifetime", "What local-path information remains after a local tile update?", "Only visited-ID metadata and current state remain; no local path tape or later tile weights are retained."),
    ("T03", 19, "text_predictor_supervision", "Is the predictor trained against its own predicted corrections?", "No. It is trained only against real anchor corrections."),
    ("T03", 20, "text_anchor_freeze_commit", "When are anchor updates committed, and how are repeated visits to one tile handled?", "Only after all corrections exist; repeated visits are summed before one tile commit."),
    ("T03", 21, "text_anchor_frame_contents", "Do frozen anchor frames retain U or V weights?", "No. They retain pulse features, latent/head activations and unmodulated transform output, but no tile weights."),
    ("T03", 22, "text_history_validity", "What bounds history capacity and the age or version lag of its records?", "At most eight demonstrations; records expire after eight step indices or more than one version of lag."),
    ("T03", 23, "text_anchor_only_behavior", "How does anchor-only mode handle non-anchor steps, the predictor and history?", "It skips non-anchor steps, leaves the predictor unchanged and uses zero history."),
    ("T03", 25, "text_tile_validation", "What tile-file schema, size cap and reload checks are documented?", "Schema-2 versioned JSON, capped at 8 KiB; reload checks schema, identity, shape, trits and checksum.", None, "Old versions"),
    ("T03", 25, "text_tile_file_bound", "What is the upper bound on versioned tile files after completed_steps?", "At most 7 + 3 * completed_steps tile files.", None, "Old versions"),
    ("T03", 25, "text_checksum_scope", "Does FNV-1a establish cryptographic authenticity of tile files?", "No. It detects accidental corruption, not malicious edits or cryptographic authenticity.", "Old versions", "The checksum"),
    ("T03", 25, "text_tile_publication", "How is a tile commit published to disk?", "Using a synced temporary file followed by rename.", None, "Old versions"),
    ("T03", 26, "text_parameter_payload_counts", "What total, shared and largest-tile coefficient counts are documented?", "19,776 total coefficients; 8,992 shared (2,248 packed bytes); largest tile 1,664 (416 packed bytes)."),
    ("T03", 26, "text_transition_array_order", "In what order are tile transition counts stored?", "Transform, curve, router, pulse context, pulse heads."),
    ("T03", 27, "text_residency_claim_scope", "Does peak_live_tiles = 1 imply one allocation or operating-system page eviction?", "No. It bounds one owned decoded tile; serialization copies and other state remain, and OS pages are not evicted."),
    ("T03", 28, "text_stop_semantics", "When is STOP checked, and what happens to final evaluation after a stop?", "STOP is checked at each step opportunity. A stopped run skips final evaluation, writes a snapshot and exits."),
    ("T03", 28, "text_training_timing_scope", "Does training duration include disk I/O and progress records?", "Yes. Initialization and evaluations are timed separately; disk counters include all phases."),
    ("T03", 31, "text_data_validation", "What size and encoding constraints apply to the small text files?", "Files are capped at 64 KiB, UTF-8 validated and document-separated; identity uses byte count and FNV-1a."),
    ("T03", 31, "text_evaluation_mutation", "How is history and weight mutation handled during the documented evaluation?", "History resets for every example, weights never update, and shared identity and tile versions are checked before/after."),
    ("T04", 1, "kernel_selection_lifecycle", "When is HEXWEAVE_KERNEL resolved, and what happens to unsupported explicit AVX2 requests?", "It is resolved once per process before dispatch. Later changes have no effect; unsupported explicit AVX2 fails.", None, "Selected AVX2"),
    ("T04", 1, "kernel_small_shape_fallback", "When does selected AVX2 dispatch fall back directly to scalar?", "For forward operations with fewer than eight rows or transpose operations with fewer than eight columns.", "Selected AVX2"),
    ("T04", 2, "kernel_accumulation_order", "How do the eight AVX2 output lanes preserve scalar reduction order?", "Forward keeps column order and transpose keeps row order, with no horizontal reassociation or FMA."),
    ("T04", 4, "packed_input_contract", "What packed length and trit codes does from_packed accept?", "Exactly ceil(rows*cols/4) bytes; codes 0/1/2 represent -1/0/+1, with zero unused high bits."),
    ("T04", 4, "packed_column_scope", "Does column decode the entire packed matrix?", "No. It returns unscaled trits for one selected column without decoding the rest or loading model-wide caches."),
    ("T04", 7, "kernel_update_specialization", "Why was the threshold-one packed-byte specialization rejected?", "It regressed measured runtime. The original general update remains the rule for every threshold."),
    ("T01", 3, "unknown_pulse_learning_rate", "What fixed numerical learning rate is specified in this excerpt?", None),
    ("T02", 13, "unknown_owner_poll_period", "What exact RSS polling interval in milliseconds is stated here?", None),
    ("T02", 8, "unknown_corpus_document_count", "How many documents does the authored fixture corpus contain?", None),
    ("T04", 1, "unknown_cpu_model", "What exact CPU product model is used for these kernel measurements?", None, None, "Selected AVX2"),
    ("T03", 25, "unknown_serialization_rate", "What measured serialization throughput in bytes per second is given?", None, None, "Old versions"),
    ("T03", 22, "unknown_history_compressor", "Which compression algorithm is used for stored history records?", None),
    ("T03", 27, "unknown_measured_peak_rss", "What measured peak process RSS in bytes is reported in this excerpt?", None),
    ("T03", 10, "unknown_optimizer_betas", "What Adam beta values are specified for the pulse matrices?", None),
]

DEVELOPMENT = [
    ("D01", 8, "benchmark_manifest_paths", "Where are relative artifact input paths resolved, and is the supplied template ready to run?", "Against the manifest directory. The supplied template is intentionally incomplete and must fail."),
    ("D01", 11, "benchmark_endpoint_access", "Which URL hosts are accepted, and can a DNS hostname be used?", "Only explicit numeric loopback hosts, such as 127.0.0.1 or [::1]. DNS hostnames are unsupported."),
    ("D01", 11, "benchmark_http_request", "Which endpoint and streaming mode are used by the benchmark adapter?", "One non-streaming /v1/completions request at a time."),
    ("D01", 14, "benchmark_dataset_schema", "What fields are required in each benchmark dataset row?", "Exactly id, prompt and expected in UTF-8 JSONL; expected may be null."),
    ("D01", 12, "benchmark_network_failure", "What happens when one HTTP request fails, and what socket does its deadline close?", "The failed request is persisted and ends the run without retry; the deadline closes only that request's socket."),
    ("D01", 17, "benchmark_wall_scope", "Does the HTTP execution wall-time budget include artifact hashing during input validation?", "No. It starts after input validation; streaming artifact hashes are excluded from inference time.", "The run's wall-time"),
    ("D01", 15, "benchmark_endpoint_binding", "Can the generic HTTP protocol prove the server loaded the pinned artifact bytes?", "No. Artifact/endpoint binding is unverified, and results set leaderboard_eligible=false.", None, "Direct backend"),
    ("D02", 3, "benchmark_latency_scope", "Is complete HTTP request latency the same measurement as first-token latency or pure decode speed?", "No. The adapter measures complete request latency; it does not measure first-token latency or pure decode speed."),
    ("D02", 1, "benchmark_validation_order", "Which module validates the request, and does validation precede HTTP or output creation?", "manifest.py validates before any HTTP request or output creation."),
    ("D03", 1, "benchmark_software_fixture", "Does tiny.jsonl establish language-model quality?", "No. It exercises parsing and result schemas; it is not a quality benchmark or capability evidence."),
    ("D01", 14, "unknown_benchmark_example_count", "How many benchmark examples are present in the user's dataset?", None),
    ("D02", 3, "unknown_benchmark_memory_measurement", "What measured server peak VRAM in bytes is reported here?", None),
]


def make_rows(specs, split):
    rows = []
    for number, spec in enumerate(specs, 1):
        source, paragraph, family, question, answer, *bounds = spec
        start, stop = (bounds + [None, None])[:2]
        evidence, begin, end = excerpt(source, paragraph, start, stop)
        citation = f"{source}:{begin}-{end}"
        prompt = f"Use evidence only; otherwise UNKNOWN.\nSOURCE {citation}\n{evidence}\nQ: {question}\nA: "
        completion = encoded({"answer": answer if answer is not None else "UNKNOWN",
                              "citation": citation if answer is not None else None}) + "\n"
        sizes = {"question": len(question.encode()), "evidence": len(evidence.encode()),
                 "completion": len(completion.encode()), "prompt": len(prompt.encode())}
        sizes["envelope"] = sizes["prompt"] - sizes["question"] - sizes["evidence"]
        sizes["sequence"] = sizes["prompt"] + sizes["completion"]
        if any(sizes[key] > cap for key, cap in (("question", 144), ("evidence", 608),
                                                ("completion", 192), ("envelope", 80), ("sequence", 1024))):
            raise ValueError(f"byte bound {split} {number}: {sizes}")
        if SOURCES[source][0] != split:
            raise ValueError("source family crossed partition")
        if RAW[source][begin:end].decode("utf-8") != evidence:
            raise ValueError("source span mismatch")
        rows.append({"id": f"{split}-{number:03d}", "split": split, "fact_family_id": family,
                     "source_id": source, "source_sha256": sha(RAW[source]),
                     "evidence_start_byte": begin, "evidence_end_byte": end, "evidence": evidence,
                     "question": question, "answerable": answer is not None,
                     "answer": answer if answer is not None else "UNKNOWN",
                     "citation": citation if answer is not None else None,
                     "prompt": prompt, "completion": completion, "utf8_bytes": sizes,
                     "authoring": "handwritten-source-grounded-qa-v1"})
    return rows


def main():
    global RAW, TEXT
    pins = {v["path"]: v for v in json.loads(PIN.read_text())["candidate_files"]}
    RAW, TEXT, sources = {}, {}, []
    for key, (partition, relative) in SOURCES.items():
        destination = HERE / "sources" / relative
        raw = (destination if destination.exists() else DRIVE / "stage" / relative).read_bytes()
        if sha(raw) != pins[relative]["sha256"]:
            raise ValueError(f"pinned T1 source changed: {relative}")
        RAW[key], TEXT[key] = raw, raw.decode("utf-8")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        sources.append({"source_id": key, "partition": partition, "original_path": relative,
                        "snapshot_path": f"sources/{relative}", "sha256": sha(raw), "bytes": len(raw)})
    train, dev = make_rows(TRAIN, "train"), make_rows(DEVELOPMENT, "development")
    if {r["fact_family_id"] for r in train} & {r["fact_family_id"] for r in dev}:
        raise ValueError("fact family crossed partition")
    if len({r["question"] for r in train + dev}) != len(train + dev):
        raise ValueError("duplicated authored question")
    artifacts = {}
    for name, rows in (("train.jsonl", train), ("development.jsonl", dev)):
        raw = ("\n".join(map(encoded, rows)) + "\n").encode("utf-8")
        (HERE / name).write_bytes(raw)
        artifacts[name] = {"sha256": sha(raw), "bytes": len(raw), "rows": len(rows),
                           "answerable": sum(r["answerable"] for r in rows),
                           "abstention": sum(not r["answerable"] for r in rows),
                           "fact_families": len({r["fact_family_id"] for r in rows}),
                           "unique_evidence_spans": len({(r["source_id"], r["evidence_start_byte"], r["evidence_end_byte"]) for r in rows}),
                           "max_utf8_bytes": {k: max(r["utf8_bytes"][k] for r in rows) for k in rows[0]["utf8_bytes"]},
                           "total_supervised_completion_bytes": sum(r["utf8_bytes"]["completion"] for r in rows)}
    manifest = {"schema": "hexweave-docs-qa-v1", "dataset_version": "t1-snapshot-d0-v1",
                "status": "authored-unreviewed", "model_trained": False,
                "source_as_of": "T1 source snapshot, 2026-10-24; not a claim about current code",
                "source_pin_manifest_sha256": sha(PIN.read_bytes()), "sources": sources,
                "artifacts": artifacts, "prompt_template": "Use evidence only; otherwise UNKNOWN.\nSOURCE {source_id}:{start}-{end}\n{evidence}\nQ: {question}\nA: ",
                "completion_format": "compact UTF-8 JSON with ordered answer,citation keys and one terminating newline",
                "abstention_completion": {"answer": "UNKNOWN", "citation": None},
                "limits": {"envelope_bytes": 80, "question_bytes": 144, "evidence_bytes": 608,
                           "completion_bytes": 192, "sequence_positions": 1024},
                "training_target_scope": "all completion UTF-8 bytes including JSON/citation/terminating newline; exclude prompt and JSONL metadata; stop generation on newline or 192 bytes",
                "external_sources_fetched": False, "final_evaluation_authored": False,
                "provenance": "Handwritten QA by /root/review_contracts; exact excerpts selected from hashed local snapshots. No synthetic paraphrase expansion.",
                "prohibited_evaluation_fact_overlap": ["shared generic routing/ternary concepts already present in train",
                    "text-vs-FP32 comparisons whose answers are already in training source",
                    "generic no-quality/no-scaling conclusions", "any paraphrase of an existing train/development fact_family_id"],
                "review_requirements": ["independent semantic/source support review", "confirm fact-family partition separation",
                    "independent final evaluation authorship; do not use development as final evaluation"]}
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "sources": len(sources), "artifacts": artifacts}, indent=2))


if __name__ == "__main__":
    main()
