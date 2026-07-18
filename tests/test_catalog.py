from __future__ import annotations

import copy
import json
import unittest

from scripts.catalog import (
    CATALOG_PATH,
    README_PATH,
    REPO_ROOT,
    load_catalog,
    render_readme,
    validate_catalog,
)


class CatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_catalog()

    def test_catalog_is_valid(self) -> None:
        self.assertEqual(validate_catalog(self.catalog), [])

    def test_schema_is_valid_json(self) -> None:
        schema_path = REPO_ROOT / "data" / "resources.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")

    def test_generated_readme_regions_are_current(self) -> None:
        current = README_PATH.read_text(encoding="utf-8")
        self.assertEqual(render_readme(self.catalog, current), current)

    def test_duplicate_id_is_rejected(self) -> None:
        catalog = copy.deepcopy(self.catalog)
        catalog["resources"][1]["id"] = catalog["resources"][0]["id"]
        errors = validate_catalog(catalog, check_local_paths=False)
        self.assertTrue(any("Duplicate resource id" in error for error in errors))

    def test_missing_local_path_is_rejected(self) -> None:
        catalog = copy.deepcopy(self.catalog)
        resource = next(item for item in catalog["resources"] if item.get("inference"))
        resource["inference"]["local"][0]["url"] = "inference/not-a-real-file.py"
        errors = validate_catalog(catalog)
        self.assertTrue(any("does not exist" in error for error in errors))

    def test_required_link_group_is_rejected_when_missing(self) -> None:
        catalog = copy.deepcopy(self.catalog)
        del catalog["resources"][0]["paper_links"]
        errors = validate_catalog(catalog, check_local_paths=False)
        self.assertTrue(any("paper_links is required" in error for error in errors))

    def test_inference_order_must_cover_inference_resources(self) -> None:
        catalog = copy.deepcopy(self.catalog)
        catalog["inference_order"].pop()
        errors = validate_catalog(catalog, check_local_paths=False)
        self.assertTrue(any("inference_order is missing" in error for error in errors))

    def test_unallowlisted_shared_url_is_rejected(self) -> None:
        catalog = copy.deepcopy(self.catalog)
        first_url = catalog["resources"][0]["paper_links"][0]["url"]
        catalog["resources"][1]["paper_links"][0]["url"] = first_url
        errors = validate_catalog(catalog, check_local_paths=False)
        self.assertTrue(any("not allowlisted" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
