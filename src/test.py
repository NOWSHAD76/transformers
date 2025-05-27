# from transformers.models.gemma3_with_talker.configuration_gemma3_with_talker import (
#     Gemma3WithTalkerConfig,
# )
# from transformers.models.gemma3_with_talker.modeling_gemma3_with_talker import (
#     Gemma3WithTalkerForConditionalGeneration,
# )
# from transformers.models.gemma3_with_talker.configuration_gemma3_with_talker import *
# from transformers import AutoConfig, AutoModel

# from loguru import logger
# from transformers import AutoConfig

# # configs = {
# #     "gemma3_thinker": Gemma3ThinkerConfig,
# #     "gemma3_talker": Gemma3TalkerConfig,
# #     "gemma3_token2wav": Gemma3Token2WavConfig,
# #     "gemma3_with_talker": Gemma3WithTalkerConfig,
# #     "gemma3_with_talker_text": Gemma3WithTalkerTextConfig,
# #     "gemma3_bigvgan": Gemma3BigVGANConfig,
# #     "gemma3_dit": Gemma3DiTConfig,
# # }
# # for name, conf in configs.items():
# #     AutoConfig.register(name, conf)

# # AutoConfig.register("gemma3_with_talker", Gemma3WithTalkerConfig)
# # AutoModel.register(
# #     Gemma3WithTalkerConfig, Gemma3PreTrainedModelForConditionalGeneration
# # )
# import json

# # with open("/data1/nowshad/models/gemma3_talker_base/config.json", "r") as f:
# #     config = json.load(f)
# model = Gemma3WithTalkerForConditionalGeneration.from_pretrained(
#     "/data1/nowshad/models/gemma3_talker_base",
#     device_map="auto",
#     # config=config,
# )
# # model.to("cpu")
# print(model)
# logger.info("Model loaded")
# messages = [
#     {
#         "role": "system",
#         "content": [{"type": "text", "text": "You are a helpful assistant."}],
#     },
#     {
#         "role": "user",
#         "content": [
#             {"type": "text", "text": "What is your name?"},
#         ],
#     },
# ]
# chat_template = "{{ bos_token }}\n{%- if messages[0]['role'] == 'system' -%}\n    {%- if messages[0]['content'] is string -%}\n        {%- set first_user_prefix = messages[0]['content'] + '\n\n' -%}\n    {%- else -%}\n        {%- set first_user_prefix = messages[0]['content'][0]['text'] + '\n\n' -%}\n    {%- endif -%}\n    {%- set loop_messages = messages[1:] -%}\n{%- else -%}\n    {%- set first_user_prefix = \"\" -%}\n    {%- set loop_messages = messages -%}\n{%- endif -%}\n{%- for message in loop_messages -%}\n    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}\n        {{ raise_exception(\"Conversation roles must alternate user/assistant/user/assistant/...\") }}\n    {%- endif -%}\n    {%- if (message['role'] == 'assistant') -%}\n        {%- set role = \"model\" -%}\n    {%- else -%}\n        {%- set role = message['role'] -%}\n    {%- endif -%}\n    {{ '<start_of_turn>' + role + '\n' + (first_user_prefix if loop.first else \"\") }}\n    {%- if message['content'] is string -%}\n        {{ message['content'] | trim }}\n    {%- elif message['content'] is iterable -%}\n        {%- for item in message['content'] -%}\n            {%- if item['type'] == 'image' -%}\n                {{ '<start_of_image>' }}\n            {%- elif item['type'] == 'text' -%}\n                {{ item['text'] | trim }}\n            {%- endif -%}\n        {%- endfor -%}\n    {%- else -%}\n        {{ raise_exception(\"Invalid content type\") }}\n    {%- endif -%}\n    {{ '<end_of_turn>\n' }}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n    {{'<start_of_turn>model\n'}}\n{%- endif -%}\n"
# processor = Gemma3Processor.from_pretrained(
#     "/data1/nowshad/models/gemma3_talker_base", chat_template=chat_template
# )
# inputs = processor.apply_chat_template(
#     messages,
#     tokenize=True,
#     return_dict=True,
#     return_tensors="pt",
#     add_generation_prompt=True,
# )
# logger.debug(f"Inputs {inputs}")
# # inputs["input_ids"] = inputs["input_ids"].to("meta")
# output = model.generate(**inputs, max_new_tokens=50, cache_implementation="static")
# print(processor.decode(output[0], skip_special_tokens=True))
