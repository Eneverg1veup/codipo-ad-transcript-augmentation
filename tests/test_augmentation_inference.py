from __future__ import annotations

import unittest

from codipo_release.augmentation_inference.generate_adapter_augmentations import (
    InferenceSettings,
)
from codipo_release.augmentation_inference.prompts import build_inference_messages


class AugmentationInferenceTests(unittest.TestCase):
    def test_final_generation_defaults(self) -> None:
        settings = InferenceSettings("base", "adapter", "image", "source", "output")
        self.assertEqual(settings.augmentations_per_source, 2)
        self.assertEqual(settings.temperature, 1.2)
        self.assertEqual(settings.max_new_tokens, 320)

    def test_chat_roles_match_executed_path(self) -> None:
        messages = build_inference_messages(1, "source")
        self.assertEqual([item["role"] for item in messages], ["system", "user"])
        self.assertEqual(messages[1]["content"][0], {"type": "image"})


if __name__ == "__main__":
    unittest.main()
