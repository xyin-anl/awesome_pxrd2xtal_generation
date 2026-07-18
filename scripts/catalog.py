#!/usr/bin/env python3
"""Validate the canonical resource catalog and render README tables."""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = REPO_ROOT / "data" / "resources.json"
README_PATH = REPO_ROOT / "README.md"

CATEGORIES = {"core_solver", "pipeline_module", "dataset", "utility"}
REQUIRED_LINK_GROUPS = {
    "core_solver": ("paper_links", "artifact_links"),
    "pipeline_module": ("paper_links", "artifact_links"),
    "dataset": ("resource_links",),
    "utility": ("resource_links",),
}
LINK_ROLES = {
    "paper",
    "code",
    "data",
    "model",
    "docs",
    "tool",
    "inference",
    "resource",
}
INFERENCE_FIELDS = ("local", "cloud", "support", "environment")

ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class CatalogError(ValueError):
    """Raised when catalog validation fails."""


def load_catalog(path: Path = CATALOG_PATH) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CatalogError(f"Catalog not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CatalogError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise CatalogError("Catalog root must be an object")
    return data


def validate_catalog(
    catalog: dict[str, Any], *, check_local_paths: bool = True
) -> list[str]:
    errors: list[str] = []

    if catalog.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if catalog.get("$schema") != "./resources.schema.json":
        errors.append("$schema must be ./resources.schema.json")
    _validate_date(catalog.get("last_updated"), "last_updated", errors)

    website_url = catalog.get("website_url")
    if not isinstance(website_url, str) or not website_url.startswith("https://"):
        errors.append("website_url must be an https URL")

    inference_order = catalog.get("inference_order", [])
    if not isinstance(inference_order, list) or not all(
        isinstance(resource_id, str) for resource_id in inference_order
    ):
        errors.append("inference_order must be an array of resource IDs")
        inference_order = []
    if len(set(inference_order)) != len(inference_order):
        errors.append("inference_order contains duplicates")

    shared_urls = catalog.get("shared_urls", [])
    if not isinstance(shared_urls, list) or not all(
        isinstance(url, str) for url in shared_urls
    ):
        errors.append("shared_urls must be an array of URL strings")
        shared_urls = []
    shared_url_set = set(shared_urls)
    if len(shared_url_set) != len(shared_urls):
        errors.append("shared_urls contains duplicates")
    for url in shared_urls:
        if not url.startswith("https://"):
            errors.append(f"shared_urls entries must use https: {url}")

    resources = catalog.get("resources")
    if not isinstance(resources, list) or not resources:
        errors.append("resources must be a non-empty array")
        return errors

    ids: list[str] = []
    names: list[str] = []
    inference_ids: list[str] = []
    external_url_owners: dict[str, set[str]] = {}

    for index, resource in enumerate(resources):
        location = f"resources[{index}]"
        if not isinstance(resource, dict):
            errors.append(f"{location} must be an object")
            continue

        resource_id = resource.get("id")
        name = resource.get("name")
        category = resource.get("category")
        if not isinstance(resource_id, str) or not ID_RE.fullmatch(resource_id):
            errors.append(f"{location}.id must match {ID_RE.pattern}")
            resource_id = location
        else:
            ids.append(resource_id)
        if not isinstance(name, str) or not name.strip():
            errors.append(f"{location}.name must be a non-empty string")
        else:
            names.append(name.casefold())
        if category not in CATEGORIES:
            errors.append(f"{location}.category must be one of {sorted(CATEGORIES)}")
            continue

        for field in _required_fields(category):
            value = resource.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{location}.{field} must be a non-empty string")
            elif "|" in value or "\n" in value:
                errors.append(f"{location}.{field} cannot contain table delimiters/newlines")

        for group_name in REQUIRED_LINK_GROUPS[category]:
            if group_name not in resource:
                errors.append(f"{location}.{group_name} is required")

        artifact_note = resource.get("artifact_note")
        if isinstance(artifact_note, str) and ("|" in artifact_note or "\n" in artifact_note):
            errors.append(
                f"{location}.artifact_note cannot contain table delimiters/newlines"
            )

        _validate_date(resource.get("verified_at"), f"{location}.verified_at", errors)

        link_groups = _link_groups_for(resource)
        seen_resource_urls: set[str] = set()
        for group_name, links in link_groups:
            if not isinstance(links, list):
                errors.append(f"{location}.{group_name} must be an array")
                continue
            for link_index, link in enumerate(links):
                link_location = f"{location}.{group_name}[{link_index}]"
                if not isinstance(link, dict):
                    errors.append(f"{link_location} must be an object")
                    continue
                label = link.get("label")
                url = link.get("url")
                role = link.get("role")
                if not isinstance(label, str) or not label.strip():
                    errors.append(f"{link_location}.label must be non-empty")
                elif "|" in label or "\n" in label:
                    errors.append(
                        f"{link_location}.label cannot contain table delimiters/newlines"
                    )
                if role not in LINK_ROLES:
                    errors.append(
                        f"{link_location}.role must be one of {sorted(LINK_ROLES)}"
                    )
                if not isinstance(url, str) or not url.strip():
                    errors.append(f"{link_location}.url must be non-empty")
                    continue
                if url in seen_resource_urls:
                    errors.append(f"{location} repeats link URL: {url}")
                seen_resource_urls.add(url)

                if _is_external_url(url):
                    if urlparse(url).scheme != "https":
                        errors.append(f"{link_location}.url must use https: {url}")
                    external_url_owners.setdefault(url, set()).add(str(resource_id))
                else:
                    if url.startswith("/") or ".." in Path(url).parts:
                        errors.append(f"{link_location}.url must be repository-relative: {url}")
                    elif check_local_paths and not (REPO_ROOT / url).exists():
                        errors.append(f"{link_location}.url does not exist: {url}")

        inference = resource.get("inference")
        if inference is not None:
            inference_ids.append(str(resource_id))
            if category != "core_solver":
                errors.append(f"{location}.inference is only valid for core_solver entries")
            if not isinstance(inference, dict):
                errors.append(f"{location}.inference must be an object")
            else:
                missing = set(INFERENCE_FIELDS) - set(inference)
                if missing:
                    errors.append(
                        f"{location}.inference is missing fields: {sorted(missing)}"
                    )
                unknown = set(inference) - set(INFERENCE_FIELDS)
                if unknown:
                    errors.append(
                        f"{location}.inference has unknown fields: {sorted(unknown)}"
                    )

    _append_duplicate_errors(ids, "resource id", errors)
    _append_duplicate_errors(names, "resource name", errors)

    if set(inference_order) != set(inference_ids):
        missing = sorted(set(inference_ids) - set(inference_order))
        extra = sorted(set(inference_order) - set(inference_ids))
        if missing:
            errors.append(f"inference_order is missing resource IDs: {missing}")
        if extra:
            errors.append(f"inference_order has unknown resource IDs: {extra}")

    duplicate_urls = {
        url: owners
        for url, owners in external_url_owners.items()
        if len(owners) > 1 and url not in shared_url_set
    }
    for url, owners in sorted(duplicate_urls.items()):
        errors.append(
            f"URL is shared by multiple resources but not allowlisted: {url} "
            f"({', '.join(sorted(owners))})"
        )

    unused_shared = shared_url_set - {
        url for url, owners in external_url_owners.items() if len(owners) > 1
    }
    for url in sorted(unused_shared):
        errors.append(f"shared_urls contains an unused URL: {url}")

    return errors


def render_generated_regions(catalog: dict[str, Any]) -> dict[str, str]:
    resources = catalog["resources"]
    return {
        "core-solvers": _render_core(resources),
        "pipeline-modules": _render_modules(resources),
        "datasets": _render_datasets(resources),
        "utilities": _render_utilities(resources),
        "inference": _render_inference(resources, catalog["inference_order"]),
    }


def render_readme(catalog: dict[str, Any], current: str) -> str:
    rendered = current
    for region_name, body in render_generated_regions(catalog).items():
        start = f"<!-- BEGIN GENERATED: {region_name} -->"
        end = f"<!-- END GENERATED: {region_name} -->"
        pattern = re.compile(
            rf"({re.escape(start)}\n).*?(\n{re.escape(end)})", re.DOTALL
        )
        if len(pattern.findall(rendered)) != 1:
            raise CatalogError(f"README must contain exactly one {region_name} region")
        rendered = pattern.sub(rf"\g<1>{body}\g<2>", rendered)
    return rendered


def write_readme(catalog: dict[str, Any]) -> None:
    current = README_PATH.read_text(encoding="utf-8")
    rendered = render_readme(catalog, current)
    README_PATH.write_text(rendered, encoding="utf-8")


def check_readme(catalog: dict[str, Any]) -> list[str]:
    current = README_PATH.read_text(encoding="utf-8")
    rendered = render_readme(catalog, current)
    if rendered == current:
        return []
    diff = "".join(
        difflib.unified_diff(
            current.splitlines(keepends=True),
            rendered.splitlines(keepends=True),
            fromfile="README.md (current)",
            tofile="README.md (generated)",
        )
    )
    return ["README generated regions are stale:\n" + diff]


def _required_fields(category: str) -> tuple[str, ...]:
    common = ("id", "name", "category", "verified_at")
    fields = {
        "core_solver": (
            "year",
            "inputs_target",
            "method",
            "reported_performance",
        ),
        "pipeline_module": ("year", "task", "method", "reported_result"),
        "dataset": ("size", "sim_exp", "format", "notes"),
        "utility": ("task", "notes"),
    }
    return common + fields[category]


def _link_groups_for(resource: dict[str, Any]) -> list[tuple[str, Any]]:
    groups = [
        (name, resource.get(name, []))
        for name in ("paper_links", "artifact_links", "resource_links")
        if name in resource
    ]
    inference = resource.get("inference")
    if isinstance(inference, dict):
        groups.extend(
            (f"inference.{name}", inference.get(name, []))
            for name in INFERENCE_FIELDS
        )
    return groups


def _render_core(resources: list[dict[str, Any]]) -> str:
    lines = [
        "| Model | Year | Inputs / Target | Method / Architecture | Reported Performance† | Paper | Implementation / Data |",
        "|-------|------|-----------------|-----------------------|------------------------|-------|-----------------------|",
    ]
    for item in _of_category(resources, "core_solver"):
        lines.append(
            _row(
                f"**{item['name']}**",
                item["year"],
                item["inputs_target"],
                item["method"],
                item["reported_performance"],
                _render_links(item.get("paper_links", [])),
                _render_links_and_note(
                    item.get("artifact_links", []), item.get("artifact_note")
                ),
            )
        )
    return "\n".join(lines)


def _render_modules(resources: list[dict[str, Any]]) -> str:
    lines = [
        "| Resource | Year | Task | Method | Reported Result / Note | Paper | Code / Data |",
        "|----------|------|------|--------|------------------------|-------|-------------|",
    ]
    for item in _of_category(resources, "pipeline_module"):
        lines.append(
            _row(
                f"**{item['name']}**",
                item["year"],
                item["task"],
                item["method"],
                item["reported_result"],
                _render_links(item.get("paper_links", [])),
                _render_links_and_note(
                    item.get("artifact_links", []), item.get("artifact_note")
                ),
            )
        )
    return "\n".join(lines)


def _render_datasets(resources: list[dict[str, Any]]) -> str:
    lines = [
        "| Dataset / Benchmark | Size | Sim / Exp | Format | Notes / Use | Link |",
        "|---------------------|------|-----------|--------|-------------|------|",
    ]
    for item in _of_category(resources, "dataset"):
        lines.append(
            _row(
                f"**{item['name']}**",
                item["size"],
                item["sim_exp"],
                item["format"],
                item["notes"],
                _render_links(item.get("resource_links", [])),
            )
        )
    return "\n".join(lines)


def _render_utilities(resources: list[dict[str, Any]]) -> str:
    lines = [
        "| Tool | Task | Notes | Link |",
        "|------|------|-------|------|",
    ]
    for item in _of_category(resources, "utility"):
        lines.append(
            _row(
                f"**{item['name']}**",
                item["task"],
                item["notes"],
                _render_links(item.get("resource_links", [])),
            )
        )
    return "\n".join(lines)


def _render_inference(
    resources: list[dict[str, Any]], inference_order: list[str]
) -> str:
    lines = [
        "| Model | Local Inference | Cloud Inference | Utils/Support | Environment |",
        "|-------|-----------------|-----------------|---------------|-------------|",
    ]
    resources_by_id = {item["id"]: item for item in resources}
    for resource_id in inference_order:
        item = resources_by_id[resource_id]
        inference = item["inference"]
        lines.append(
            _row(
                f"**{item['name']}**",
                *(
                    _render_links(inference.get(field, [])) or "N/A"
                    for field in INFERENCE_FIELDS
                ),
            )
        )
    return "\n".join(lines)


def _of_category(
    resources: list[dict[str, Any]], category: str
) -> list[dict[str, Any]]:
    return [item for item in resources if item.get("category") == category]


def _row(*cells: str) -> str:
    return "| " + " | ".join(cells) + " |"


def _render_links(links: list[dict[str, Any]]) -> str:
    rendered = []
    for link in links:
        label = link["label"]
        if link.get("code_label"):
            label = f"`{label}`"
        rendered.append(f"[{label}]({link['url']})")
    return ", ".join(rendered)


def _render_links_and_note(
    links: list[dict[str, Any]], note: str | None
) -> str:
    link_text = _render_links(links)
    if link_text and note:
        return f"{link_text}, {note}"
    return link_text or note or "N/A"


def _validate_date(value: Any, field: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not DATE_RE.fullmatch(value):
        errors.append(f"{field} must be an ISO date (YYYY-MM-DD)")
        return
    try:
        date.fromisoformat(value)
    except ValueError:
        errors.append(f"{field} is not a valid date: {value}")


def _append_duplicate_errors(values: list[str], label: str, errors: list[str]) -> None:
    for value, count in Counter(values).items():
        if count > 1:
            errors.append(f"Duplicate {label}: {value}")


def _is_external_url(url: str) -> bool:
    return url.startswith("http://") or url.startswith("https://")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "render", "check"))
    args = parser.parse_args(argv)

    try:
        catalog = load_catalog()
        errors = validate_catalog(catalog)
        if args.command == "check" and not errors:
            errors.extend(check_readme(catalog))
        if errors:
            print("Catalog validation failed:", file=sys.stderr)
            for error in errors:
                print(f"- {error}", file=sys.stderr)
            return 1
        if args.command == "render":
            write_readme(catalog)
            print(f"Rendered {README_PATH.relative_to(REPO_ROOT)}")
        else:
            print(
                f"Catalog is valid: {len(catalog['resources'])} resources",
                file=sys.stdout,
            )
        return 0
    except CatalogError as exc:
        print(f"Catalog error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
