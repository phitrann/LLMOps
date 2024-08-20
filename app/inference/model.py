"""
https://github.com/huggingface/transformers/blob/main/src/transformers/generation/streamers.py
"""
from enum import Enum
from typing import Any, Iterator, AsyncIterator, Union
from threading import Thread
from queue import Empty
import asyncio
from functools import lru_cache

from transformers import TextIteratorStreamer
from langchain.prompts import StringPromptTemplate

from app.api.config import Settings
from app.api.utils import get_settings


class BackendType(Enum):
    TRANSFORMERS = "transformers"
    OLLAMA = "ollama"
    ONNX = "onnx"
    LLAMA_CPP = "llama.cpp"


class LLMModel:
    def __init__(self, settings: Settings = get_settings()):
        self.model_path = settings.MODEL_PATH
        self.backend_type = BackendType(settings.BACKEND_TYPE)
        self.max_new_tokens = settings.MAX_NEW_TOKENS
        self.max_tokens = settings.MAX_TOKENS
        self.load_in_8bit = settings.LOAD_IN_8BITS

        self.model = None
        self.tokenizer = None

        self.verbose = settings.VERBOSE
        self.timings = settings.TIMINGS

        self._initialize_backend()
        self.init_model()
        self.init_tokenizer()

    def _initialize_backend(self):
        if self.backend_type == BackendType.LLAMA_CPP:
            print("Running llama.cpp backend")
        elif self.backend_type == BackendType.OLLAMA:
            print("Running Ollama backend")
        elif self.backend_type == BackendType.ONNX:
            print("Running ONNX backend")
        else:
            import torch

            if torch.cuda.is_available():
                print("Running GPU backend with Torch Transformers.")
            else:
                print(
                    "GPU CUDA not found. Running CPU backend with Torch Transformers."
                )

    def init_model(self):
        if self.model is None:
            self.model = self._create_model()

    def init_tokenizer(self):
        if self.backend_type != BackendType.LLAMA_CPP and self.tokenizer is None:
            self.tokenizer = self._create_tokenizer()

    @lru_cache(maxsize=1)
    def _create_model(self):
        if self.backend_type == BackendType.LLAMA_CPP:
            from llama_cpp import Llama

            return Llama(
                model_path=self.model_path,
                n_ctx=self.max_tokens,
                n_batch=self.max_tokens,
                verbose=self.verbose,
            )
        elif self.backend_type == BackendType.OLLAMA:
            # from langchain.chat_models import ChatOllama
            # return ChatOllama(model="phi3:instruct", base_url="http://localhost:11434")
            from llama_index.llms.ollama import Ollama

            return Ollama(
                base_url="http://localhost:11434",
                model="llama3.1",
                request_timeout=60.0,
            )
        elif self.backend_type == BackendType.ONNX:
            import onnxruntime_genai as og

            return og.Model(self.model_path)
        else:  # Transformers
            from transformers import AutoModelForCausalLM
            import torch

            attn_implementation = "flash_attention_2"  # "eager"

            return AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="cuda" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.float16,
                load_in_8bit=self.load_in_8bit,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
            ).eval()

    @lru_cache(maxsize=1)
    def _create_tokenizer(self):
        if self.backend_type == BackendType.TRANSFORMERS:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(self.model_path)
        elif self.backend_type == BackendType.ONNX:
            import onnxruntime_genai as og

            return og.Tokenizer(self.model)

    def get_token_length(self, prompt: str) -> int:
        if self.backend_type == BackendType.LLAMA_CPP:
            return len(self.model.tokenize(bytes(prompt, "utf-8")))
        else:
            return len(self.tokenizer.encode(prompt))

    def get_input_token_length(
        self,
        message: str,
        chat_history: list[tuple[str, str]] = [],
        system_prompt: str = "",
    ) -> int:
        prompt = get_prompt(message, chat_history, system_prompt)
        return self.get_token_length(prompt)

    async def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1000,
        temperature: float = 0.9,
        top_p: float = 1.0,
        top_k: int = 40,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
        **kwargs: Any,
    ) -> AsyncIterator[str]:

        # Process the prompt, removing the kwargs used for prompt formatting
        prompt_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in ["chat_history", "context", "question"]
        }
        processed_prompt = self._process_prompt(prompt, **prompt_kwargs)

        # Remove prompt-specific kwargs from the general kwargs
        for key in prompt_kwargs:
            kwargs.pop(key, None)

        if self.backend_type == BackendType.LLAMA_CPP:
            result = self.model(
                processed_prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repetition_penalty,
                **kwargs,
            )
            outputs = []
            for part in result:
                text = part["choices"][0]["text"]
                outputs.append(text)
                yield "".join(outputs)
        elif self.backend_type == BackendType.OLLAMA:
            for chunk in self.model.stream(processed_prompt):
                yield chunk.content
        elif self.backend_type == BackendType.ONNX:
            # Implement ONNX streaming generation
            import onnxruntime_genai as og
            import time

            tokenizer_stream = self.tokenizer.create_stream()

            search_options = {"max_length": 1024, "temperature": 0.3}

            params = og.GeneratorParams(self.model)
            # params.try_graph_capture_with_max_batch_size(1)
            params.set_search_options(**search_options)

            input_tokens = self.tokenizer.encode(processed_prompt)
            params.input_ids = input_tokens

            generator = og.Generator(self.model, params)
            if self.verbose:
                print("Generator created")

            if self.verbose:
                print("Running generation loop ...")

            new_tokens = []
            # if self.timings:
            #     nonlocal first_token_timestamp, first
            try:
                while not generator.is_done():
                    generator.compute_logits()
                    generator.generate_next_token()
                    # if self.timings:
                    #     if first:
                    #         first_token_timestamp = time.time()
                    #         first = False

                    new_token = generator.get_next_tokens()[0]
                    print(tokenizer_stream.decode(new_token), end="", flush=True)
                    yield tokenizer_stream.decode(new_token)
                    if self.timings:
                        new_tokens.append(new_token)
            except KeyboardInterrupt:
                yield "  --control+c pressed, aborting generation--"
            finally:
                del generator
                # if self.timings:
                #     prompt_time = first_token_timestamp - started_timestamp
                #     run_time = time.time() - first_token_timestamp
                #     yield f"\nPrompt length: {len(input_tokens)}, New tokens: {len(new_tokens)}, Time to first: {(prompt_time):.2f}s, Prompt tokens per second: {len(input_tokens)/prompt_time:.2f} tps, New tokens per second: {len(new_tokens)/run_time:.2f} tps"

        else:  # Transformers
            inputs = self.tokenizer(processed_prompt, return_tensors="pt").to(
                self.model.device
            )
            streamer = TextIteratorStreamer(
                self.tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
            )
            generate_kwargs = dict(
                inputs,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                **kwargs,
            )

            generate_kwargs = (
                generate_kwargs if kwargs is None else {**generate_kwargs, **kwargs}
            )

            t = Thread(target=self.model.generate, kwargs=generate_kwargs)
            t.start()

            outputs = []
            for text in streamer:
                outputs.append(text)
                # yield "".join(outputs)
                yield text
    
    def run(
        self,
        message: str,
        chat_history: list[tuple[str, str]] = [],
        system_prompt: str = "",
        max_new_tokens: int = 1000,
        temperature: float = 0.9,
        top_p: float = 1.0,
        top_k: int = 40,
        repetition_penalty: float = 1.0,
    ) -> Iterator[str]:
        prompt = get_prompt(message, chat_history, system_prompt)
        return self.generate(
            prompt, max_new_tokens, temperature, top_p, top_k, repetition_penalty
        )

    def _process_prompt(
        self, prompt: Union[str, StringPromptTemplate], **kwargs
    ) -> str:
        if isinstance(prompt, StringPromptTemplate):
            return prompt.format(**kwargs)
        return prompt

    async def __call__(
        self,
        prompt: Union[str, StringPromptTemplate],
        stream: bool = False,
        max_new_tokens: int = 1000,
        temperature: float = 0.9,
        top_p: float = 1.0,
        top_k: int = 40,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
        **kwargs: Any,
    ) -> Union[str, Iterator[str]]:

        # Process the prompt, removing the kwargs used for prompt formatting
        prompt_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in ["chat_history", "context", "question"]
        }
        processed_prompt = self._process_prompt(prompt, **prompt_kwargs)

        # Remove prompt-specific kwargs from the general kwargs
        for key in prompt_kwargs:
            kwargs.pop(key, None)

        if self.backend_type == BackendType.LLAMA_CPP:
            return await self._generate_llama_cpp(
                processed_prompt,
                stream,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                **kwargs,
            )
        elif self.backend_type == BackendType.OLLAMA:
            return await self._generate_ollama(processed_prompt, stream)
        elif self.backend_type == BackendType.ONNX:
            # Implement ONNX generation
            pass
        else:  # Transformers
            return await self._generate_transformers(
                processed_prompt,
                stream,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
                do_sample,
                repetition_penalty,
                **kwargs,
            )

    async def _generate_llama_cpp(
        self,
        prompt,
        stream,
        max_new_tokens,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        **kwargs,
    ):
        completion_or_chunks = await asyncio.to_thread(
            self.model,
            prompt,
            stream=stream,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repetition_penalty,
            **kwargs,
        )
        if stream:

            async def chunk_generator(chunks):
                for part in chunks:
                    yield part["choices"][0]["text"]

            return chunk_generator(completion_or_chunks)
        return completion_or_chunks["choices"][0]["text"]

    async def _generate_ollama(self, prompt, stream):
        if stream:
            return await self.model.astream(prompt)
        else:
            response = await self.model.agenerate(prompt)
            return response.generations[0][0].text

    async def _generate_transformers(
        self,
        prompt,
        stream,
        max_new_tokens,
        temperature,
        top_p,
        top_k,
        do_sample,
        repetition_penalty,
        **kwargs,
    ):
        inputs = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.model.device
        )
        if stream:
            streamer = TextIteratorStreamer(
                self.tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
            )
            generation_kwargs = dict(
                input_ids=inputs,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                **kwargs,
            )
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            return streamer
        else:
            output_ids = await asyncio.to_thread(
                self.model.generate,
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                **kwargs,
            )
            return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    async def generate_text(self, prompt: str, streamer: TextIteratorStreamer):
        if self.backend_type == BackendType.ONNX:
            # Implement ONNX text generation
            pass
        else:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
                self.model.device
            )
            self.model.generate(
                input_ids, streamer=streamer, max_length=self.max_new_tokens
            )

    # async def consume_streamer(self, streamer: TextIteratorStreamer):
    #     while True:
    #         try:
    #             for token in streamer:
    #                 yield str(self.tokenizer.decode(token))
    #             break
    #         except Empty:
    #             await asyncio.sleep(2)


def get_prompt(
    message: str, chat_history: list[tuple[str, str]] = [], system_prompt: str = ""
) -> str:
    texts = [f"<|system|>\n{system_prompt}<|end|>\n"]
    for user_input, response in chat_history:
        texts.append(
            f"<|user|>\n{user_input.strip()}<|end]>\n<|assistant|>\n{response.strip()}<|end|>\n"
        )
    texts.append(f"<|user|>\n{message.strip()}<|end|>\n<|assistant|>")
    return "".join(texts)


# llm_model = LLMModel(settings.MODEL_PATH, use_onnx=settings.USE_ONNX)
