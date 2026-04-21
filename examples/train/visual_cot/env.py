from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from typing import Dict, Any, Tuple
import base64
import copy
import io
import re
import json
import numpy as np
from shapely.geometry import box

class VisualCotEnv(BaseTextEnv):
    """
    Environment for multiplication.
    """

    def __init__(
        self,
        env_config: Dict[str, Any] = {},
        extras: Dict[str, Any] = {},
    ):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        self.reward_spec = json.loads(extras["reward_spec"])
        assert "bbox" in self.reward_spec, "bbox is required in reward_spec field"
        assert "answer" in self.reward_spec, "answer is required in reward_spec field"

        self.max_turns = extras["max_turns"] if "max_turns" in extras else 5
        self.images = extras.get("images", [])
        self.bbox_found = False
        
        extra_info = json.loads(extras["extra_info"])
        width = extra_info["width"]
        height = extra_info["height"]
        self.denormalization_array = np.array([
            1000/width, 1000/height, 1000/width, 1000/height,
        ])

    def _image_to_base64_url(self, img) -> str:
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    def init(self, prompt) -> Tuple[list, Dict[str, Any]]:
        """Replace <image> placeholders with base64 image_url content parts for vLLM."""
        conversation = copy.deepcopy(prompt)
        for message in conversation:
            content = message.get("content", "")
            if not isinstance(content, str) or "<image>" not in content:
                continue
            text = content.replace("<image>\n", "").replace("<image>", "").strip()
            content_parts = []
            for img in self.images:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": self._image_to_base64_url(img)},
                })
            if text:
                content_parts.append({"type": "text", "text": text})
            message["content"] = content_parts
        return conversation, {}

    def _parse_bbox(self, action: str) -> str:
        match = re.search(r"\[\s*\d+(\.\d+)?(\s*,\s*\d+(\.\d+)?){3}\s*\]", action)
        return json.loads(match.group(0)) if match else None

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        
        reward = 0.0
        done = self.turns >= self.max_turns
        if not self.bbox_found:
            normalized_bbox = self._parse_bbox(action)
            if normalized_bbox is not None:
                bbox = np.array(normalized_bbox) * self.denormalization_array
                bbox = list(bbox)
                pred_box = box(*bbox)
                gt_box = box(*(self.reward_spec["bbox"][0]))
                iou = pred_box.intersection(gt_box).area / pred_box.union(gt_box).area
                if iou > 0.8:
                    self.bbox_found=True
                    reward = 0.8
                    feedback = "The bounding box looks correct. Now what was the answer to the original question?"
                else:
                    reward = 0.1 # Formatting reward
                    feedback = "That bounding box looks off, can you try again?"
            else:
                feedback = "Please reason and provide the bounding box coordinates to answer the question in the format `[x1, y1, x2, y2]`."
        else:
            done = True # Get 1 shot to answer question
            if self.reward_spec["answer"].lower() in action.lower(): # Very simple check
                reward = 0.2

        if done:
            return BaseTextEnvStepOutput(observations=[], reward=reward, done=True, metadata={"parsed_answer": action})

        new_obs = {"role": "user", "content": feedback}
        return BaseTextEnvStepOutput(observations=[new_obs], reward=reward, done=False, metadata={"parsed_answer": action})
