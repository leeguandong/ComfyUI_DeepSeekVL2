import os
import PIL.Image
import torch
from PIL import Image
from typing import List, Dict
from torchvision.transforms import ToPILImage
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM
from .deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor

import comfy.model_management as mm
from comfy.utils import ProgressBar
import folder_paths

script_directory = os.path.dirname(os.path.abspath(__file__))


class deepseek_vl2_model_loader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": (
                [
                    "deepseek-vl2-tiny",
                    "deepseek-vl2-small",
                    "deepseek-vl2",
                ],
                {
                    "default": "deepseek-vl2-small"
                }),
            "load_local_model": ("BOOLEAN", {"default": False}),
        }, "optional": {
            "local_model_path": ("STRING", {"default": "deepseek-vl2-small"}),
        }
        }

    RETURN_TYPES = ("DEEPSEEKVLMODEL",)
    RETURN_NAMES = ("deepseek_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "deepseek-vl2"

    def loadmodel(self, model, load_local_model, *args, **kwargs):
        mm.soft_empty_cache()
        dtype = torch.bfloat16
        device = mm.get_torch_device()

        if load_local_model:
            model_path = kwargs.get("local_model_path", "deepseek-vl2-small")
        else:
            model_path = (os.path.join(folder_paths.models_dir, "LLM"))
            checkpoint_path = os.path.join(model_path, model)
            snapshot_download(repo_id=f"deepseek-ai/{model}",
                              local_dir=checkpoint_path,
                              local_dir_use_symlinks=False)

        vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
        tokenizer = vl_chat_processor.tokenizer

        vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype
        )
        vl_gpt = vl_gpt.to(device).eval()
        deepseek_vl_model = {
            "chat_processor": vl_chat_processor,
            "model": vl_gpt,
            "tokenizer": tokenizer
        }

        return (deepseek_vl_model,)


class deepseek_vl2_inference:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "images": ("IMAGE",),
            "deepseek_vl2_model": ("DEEPSEEKVLMODEL",),
            "prompt": ("STRING", {"multiline": True, "default": "Describe the image in detail.", }),
        },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "process"
    CATEGORY = "deepseek-vl2"

    def process(self, images, deepseek_vl_model, prompt):
        mm.soft_empty_cache()
        dtype = torch.bfloat16
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        images = images.permute(0, 3, 1, 2)
        to_pil = ToPILImage()

        vl_chat_processor = deepseek_vl_model["chat_processor"]
        vl_gpt = deepseek_vl_model["model"]
        tokenizer = deepseek_vl_model["tokenizer"]

        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image>\n{prompt}",
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        # load images and prepare for inputs
        # pil_images = load_pil_images(conversation)
        # print(f"len(pil_images) = {len(pil_images)}")
        pbar = ProgressBar(len(images))
        answer_list = []
        vl_gpt.to(device)
        for img in images:
            pil_image = to_pil(img)
            prepare_inputs = vl_chat_processor.__call__(
                conversations=conversation,
                images=[pil_image],
                force_batchify=True,
                system_prompt=""
            ).to(vl_gpt.device, dtype=dtype)

            with torch.no_grad():
                inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
                past_key_values = None

                # run the model to get the response
                outputs = vl_gpt.generate(
                    # inputs_embeds=inputs_embeds[:, -1:],
                    # input_ids=prepare_inputs.input_ids[:, -1:],
                    inputs_embeds=inputs_embeds,
                    input_ids=prepare_inputs.input_ids,
                    images=prepare_inputs.images,
                    images_seq_mask=prepare_inputs.images_seq_mask,
                    images_spatial_crop=prepare_inputs.images_spatial_crop,
                    attention_mask=prepare_inputs.attention_mask,
                    past_key_values=past_key_values,

                    pad_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=512,

                    # do_sample=False,
                    # repetition_penalty=1.1,

                    do_sample=True,
                    temperature=0.4,
                    top_p=0.9,
                    repetition_penalty=1.1,

                    use_cache=True,
                )

                answer = tokenizer.decode(outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(),
                                          skip_special_tokens=False)
                # print(f"{prepare_inputs['sft_format'][0]}", answer)
                answer = answer.lstrip(" [User]\n\n")
                answer_list.append(answer)
                pbar.update(1)

        vl_gpt.to(offload_device)
        # print(f"{prepare_inputs['sft_format'][0]}", answer)
        if (len(images)) > 1:
            return (answer_list,)
        else:
            return (answer_list[0],)


NODE_CLASS_MAPPINGS = {
    "deepseek_vl2_model_loader": deepseek_vl2_model_loader,
    "deepseek_vl2_inference": deepseek_vl2_inference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "deepseek_vl2_model_loader": "DeepSeek-VL2 Model Loader",
    "deepseek_vl2_inference": "DeepSeek-VL2 Inference",
}
