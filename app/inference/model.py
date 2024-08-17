# import os
# from transformers import AutoModelForCausalLM, AutoTokenizer

# # Get the model path from environment variables or use a default path
# model_path = os.getenv("MODEL_PATH", "/models/microsoft/Phi-3-mini-4k-instruct")

# class LLMModel:
#     def __init__(self):
#         # Load the tokenizer and model with appropriate configurations
#         self.tokenizer = AutoTokenizer.from_pretrained(model_path)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_path,
#             device_map="auto",  # Automatically selects the device (GPU if available)
#             torch_dtype="auto",  # Automatically selects the appropriate dtype (float16/32)
#             trust_remote_code=True,  # Required for loading models with custom code
#         )

#     def generate(self, messages: list):
#         # Concatenate messages into a single prompt string
#         prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

#         # Tokenize the prompt
#         inputs = self.tokenizer(prompt, return_tensors="pt")

#         # Ensure the inputs are moved to the correct device (e.g., GPU if available)
#         inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

#         # Generate the response
#         outputs = self.model.generate(**inputs, max_new_tokens=512)

#         # Decode and return the generated text
#         return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# # Instantiate the model
# llm_model = LLMModel()


import os
import time
import onnxruntime_genai as og
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from queue import Empty
import asyncio
import torch

from app.api.utils import get_settings
settings = get_settings()

from queue import Empty, Queue
class RawStreamer:
    def __init__(self, timeout: float = None):
        self.q = Queue()
        self.stop_signal = None
        self.timeout = timeout

    def put(self, values):
        self.q.put(values)

    def end(self):
        self.q.put(self.stop_signal)

    def __iter__(self):
        return self

    def __next__(self):
        result = self.q.get(timeout=self.timeout)
        if result == self.stop_signal:
            raise StopIteration()
        else:
            return result

class LLMModel:
    def __init__(self, model_path, use_onnx=False):
        self.use_onnx = use_onnx
        self.model_path = model_path

        if self.use_onnx:
            self.model = og.Model(model_path)
            self.tokenizer = og.Tokenizer(self.model)
            self.tokenizer_stream = self.tokenizer.create_stream()
            self.search_options = {
                'do_sample': settings.DO_SAMPLE,
                'max_length': settings.MAX_LENGTH,
                'min_length': settings.MIN_LENGTH,
                'top_p': settings.TOP_P,
                'top_k': settings.TOP_K,
                'temperature': settings.TEMPERATURE,
                'repetition_penalty': settings.REPETITION_PENALTY,
            }
            if 'max_length' not in self.search_options:
                self.search_options['max_length'] = 2048
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype="auto",
                trust_remote_code=True,
            )

    async def generate_text(self, prompt: str, streamer: TextIteratorStreamer):
        if self.use_onnx:
            input_tokens = self.tokenizer.encode(prompt)
            params = og.GeneratorParams(self.model)
            params.set_search_options(**self.search_options)
            params.input_ids = input_tokens
            generator = og.Generator(self.model, params)
            # streamer = []
            while not generator.is_done():
                generator.compute_logits()
                generator.generate_next_token()
                new_token = generator.get_next_tokens()[0]
                # Ensure new_token is passed as a list
                yield self.tokenizer_stream.decode(new_token)
        else:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
            self.model.generate(input_ids, streamer=streamer, max_length=settings.MAX_NEW_TOKENS)

    async def consume_streamer(self, streamer: TextIteratorStreamer):
        while True:
            try:
                for token in streamer:
                    yield str(self.tokenizer.decode(token))
                break
            except Empty:
                await asyncio.sleep(0.001)

llm_model = LLMModel(settings.MODEL_PATH, use_onnx=settings.USE_ONNX)
