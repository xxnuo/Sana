from __future__ import annotations

import ast
from pathlib import Path

GPU_INFER = Path(__file__).resolve().parents[1] / "gpu_infer.py"


def _call_name(node: ast.Call) -> str:
    parts: list[str] = []
    current: ast.expr = node.func
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def test_local_fp8_backend_is_installed_before_qwen_load() -> None:
    tree = ast.parse(GPU_INFER.read_text(encoding="utf-8"), filename=str(GPU_INFER))
    build_pipeline = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_pipeline"
    )
    calls = [
        (node.lineno, _call_name(node))
        for node in ast.walk(build_pipeline)
        if isinstance(node, ast.Call)
    ]
    patch_line = min(line for line, name in calls if name == "patch_finegrained_fp8")
    qwen_load_line = min(
        line
        for line, name in calls
        if name == "Qwen3VLForConditionalGeneration.from_pretrained"
    )

    assert patch_line < qwen_load_line


def test_qwen_first_load_order_is_available_for_uma_bringup() -> None:
    source = GPU_INFER.read_text(encoding="utf-8")
    assert 'H3_LOAD_ORDER' in source
    assert '"qwen_first"' in source
    assert 'torch.cuda.empty_cache()' in source


def test_dit_and_qwen_loaders_keep_distinct_config_objects() -> None:
    tree = ast.parse(GPU_INFER.read_text(encoding="utf-8"), filename=str(GPU_INFER))
    build_pipeline = _function_node(tree, "build_pipeline")
    load_dit = next(
        node
        for node in build_pipeline.body
        if isinstance(node, ast.FunctionDef) and node.name == "_load_dit"
    )
    dit_call = next(
        node
        for node in ast.walk(load_dit)
        if isinstance(node, ast.Call) and _call_name(node) == "build_pruned_fp8_transformer"
    )
    assert isinstance(dit_call.args[1], ast.Name)
    assert dit_call.args[1].id == "dit_config"

    qwen_call = next(
        node
        for node in ast.walk(build_pipeline)
        if isinstance(node, ast.Call)
        and _call_name(node) == "Qwen3VLForConditionalGeneration.from_pretrained"
    )
    keywords = {keyword.arg: keyword.value for keyword in qwen_call.keywords}
    assert isinstance(keywords.get("config"), ast.Name)
    assert keywords["config"].id == "qwen_config"


def test_h3_component_loading_uses_an_explicit_workflow_component_list() -> None:
    tree = ast.parse(GPU_INFER.read_text(encoding="utf-8"), filename=str(GPU_INFER))
    workflow_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_name(node) == "pipe.load_components"
    ]
    assert len(workflow_calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in workflow_calls[0].keywords}
    assert "names" in keywords
    assert "workflow" not in keywords


def test_thor_loads_fp32_vaes_and_records_the_observed_parameter_dtypes() -> None:
    source = GPU_INFER.read_text(encoding="utf-8")

    assert 'vae_load_dtype = torch.float32 if _flag("H3_REQUIRE_SM110") else torch.bfloat16' in source
    assert '"vae": vae_load_dtype' in source
    assert '"audio_vae": vae_load_dtype' in source
    assert 'vae_weight_dtype = _uniform_floating_parameter_dtype(pipe.vae, "video VAE")' in source
    assert '"vae_weight_dtype": placement["vae_weight_dtype"]' in source
    assert '"audio_vae_weight_dtype": placement["audio_vae_weight_dtype"]' in source
    assert '"video_vae_autocast_dtype": os.environ.get("H3_VIDEO_VAE_AUTOCAST_DTYPE")' in source
    assert '"audio_vae_autocast_dtype": os.environ.get("H3_AUDIO_VAE_AUTOCAST_DTYPE")' in source


def test_thor_architecture_gate_runs_before_checkpoint_hash_and_model_load() -> None:
    source = GPU_INFER.read_text(encoding="utf-8")
    gate = source.index('if _flag("H3_REQUIRE_SM110"):')
    checkpoint = source.index("dit_checkpoint = _checkpoint_provenance(DIT_CHECKPOINT)")
    model_load = source.index("pipe, placement = build_pipeline")
    assert gate < checkpoint < model_load


def test_sol_attention_installer_receives_dense_suffix_policy() -> None:
    tree = ast.parse(GPU_INFER.read_text(encoding="utf-8"), filename=str(GPU_INFER))
    apply_approximations = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "apply_approximations"
    )
    patch_calls = [
        node
        for node in ast.walk(apply_approximations)
        if isinstance(node, ast.Call) and _call_name(node) == "fusion_install.patch_sol_attn"
    ]

    assert len(patch_calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in patch_calls[0].keywords}
    assert isinstance(keywords.get("dense_last_steps"), ast.Name)
    assert keywords["dense_last_steps"].id == "dense_last_steps"


def test_sol_attention_installer_receives_threshold_type() -> None:
    tree = ast.parse(GPU_INFER.read_text(encoding="utf-8"), filename=str(GPU_INFER))
    apply_approximations = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "apply_approximations"
    )
    patch_call = next(
        node
        for node in ast.walk(apply_approximations)
        if isinstance(node, ast.Call) and _call_name(node) == "fusion_install.patch_sol_attn"
    )
    keywords = {keyword.arg: keyword.value for keyword in patch_call.keywords}
    assert isinstance(keywords.get("thresh_type"), ast.Name)
    assert keywords["thresh_type"].id == "thresh_type"
    source = GPU_INFER.read_text(encoding="utf-8")
    assert 'def _sol_attn_thresh_type()' in source
    assert 'value not in ("diag", "exact")' in source
    assert 'thresh_type = _sol_attn_thresh_type()' in source
    assert '"sol_attn_thresh_type"' in source
    for name in (
        "H3_SOL_ATTN_THRESH_TYPE",
        "H3_SOL_THRESH_TYPE",
        "SOL_ATTN_THRESH_TYPE",
    ):
        assert name in source


def test_benchmark_is_written_after_output_artifact_metadata() -> None:
    source = GPU_INFER.read_text(encoding="utf-8")
    video_hash = source.index('result["video_sha256"]')
    benchmark_write = source.index('with open(result_path, "w")')
    assert video_hash < benchmark_write
    assert 'result["latents_sha256"]' in source
    assert 'result["request_plus_artifact_s"]' in source


def test_benchmark_records_launcher_provenance_digests() -> None:
    source = GPU_INFER.read_text(encoding="utf-8")
    expected = {
        '"source_snapshot_sha256": os.environ.get("H3_SOURCE_SNAPSHOT_SHA256")',
        '"runtime_source_sha256": os.environ.get("H3_RUNTIME_SOURCE_SHA256")',
        '"diffusers_commit": os.environ.get("H3_DIFFUSERS_COMMIT")',
        '"diffusers_tree_sha256": os.environ.get("H3_DIFFUSERS_TREE_SHA256")',
        '"asset_manifest": os.environ.get("H3_ASSET_MANIFEST_PATH")',
        '"asset_manifest_sha256": os.environ.get("H3_ASSET_MANIFEST_SHA256")',
    }
    assert expected <= {line.strip().rstrip(",") for line in source.splitlines()}
    assert '"runtime_min_available_mib": (' in source
    assert 'os.environ["H3_BENCH_RUNTIME_MIN_AVAILABLE_MIB"]' in source


def test_benchmark_records_fusion_installation_and_runtime_hits() -> None:
    source = GPU_INFER.read_text(encoding="utf-8")

    assert "kernel_options = lossless_kernels()" in source
    assert "fusions.reset_fusion_stats()" in source
    assert "kernel_fusion_hits = fusions.read_fusion_stats()" in source
    assert '"kernel_fusion_installation": placement["kernel_fusion_installation"]' in source
    assert '"kernel_fusion_hits": kernel_fusion_hits' in source
    assert "lossless kernel fusion was requested but did not execute" in source
    assert 'transformers_module = sys.modules.get("transformers")' in source
    assert '"transformers": getattr(transformers_module, "__version__", None)' in source


def test_benchmark_records_the_effective_dit_checkpoint_digest() -> None:
    source = GPU_INFER.read_text(encoding="utf-8")
    assert 'dit_checkpoint = _checkpoint_provenance(DIT_CHECKPOINT)' in source
    assert '"dit_checkpoint": dit_checkpoint["path"]' in source
    assert '"dit_checkpoint_size_bytes": dit_checkpoint["size_bytes"]' in source
    assert '"dit_checkpoint_sha256": dit_checkpoint["sha256"]' in source


def test_benchmark_records_the_effective_prompt_digest() -> None:
    source = GPU_INFER.read_text(encoding="utf-8")
    assert '"prompt_sha256": hashlib.sha256(args.prompt.encode("utf-8")).hexdigest()' in source
    assert '"pipeline_call_wall_clock_after_warmup_"' in source
    assert '"excluding_load_and_artifact_write"' in source
    assert 'serialized_requests = [round(value, 6) for value in requests]' in source
    assert '"request_s_median": round(_request_s_median(serialized_requests), 6)' in source


def test_prompt_can_be_supplied_without_rewriting_the_prompt_file() -> None:
    source = GPU_INFER.read_text(encoding="utf-8")
    assert 'parser.add_argument("--prompt", default=os.environ.get("H3_PROMPT") or None)' in source


def _function_node(tree: ast.AST, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _has_wraps_decorator(function: ast.FunctionDef) -> bool:
    return any(
        isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and isinstance(decorator.func.value, ast.Name)
        and decorator.func.value.id == "functools"
        and decorator.func.attr == "wraps"
        for decorator in function.decorator_list
    )


def test_transformer_wrappers_preserve_diffusers_forward_signature() -> None:
    tree = ast.parse(GPU_INFER.read_text(encoding="utf-8"), filename=str(GPU_INFER))
    instrument = _function_node(tree, "instrument")
    wrapped = next(
        node
        for node in instrument.body
        if isinstance(node, ast.FunctionDef) and node.name == "wrapped"
    )
    assert _has_wraps_decorator(wrapped)


def test_cache_context_wrapper_preserves_diffusers_forward_signature() -> None:
    cache_path = GPU_INFER.with_name("cache_line.py")
    tree = ast.parse(cache_path.read_text(encoding="utf-8"), filename=str(cache_path))
    apply_cache = _function_node(tree, "apply_cache")
    wrapped = next(
        node
        for node in apply_cache.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    assert _has_wraps_decorator(wrapped)
