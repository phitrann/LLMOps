import pytest
import httpx
import asyncio
import time


# Base URL of the LLM serving API
BASE_URL = "http://localhost:8046"


# Helper function to make an asynchronous request to the LLM API
async def make_request(message: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/chat/sse/",
            json={"session_id": "test_session", "message": message},
            timeout=None
        )
        return response


# Test Case: Measure Time To First Token (TTFT)
@pytest.mark.asyncio
async def test_time_to_first_token():
    start_time = time.time()

    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            f"{BASE_URL}/chat/sse/",
            json={"session_id": "test_session", "message": "What is the capital of France?"}
        ) as response:
            response.raise_for_status()

            async for chunk in response.aiter_text():
                if chunk.strip():
                    first_token_time = time.time()
                    ttft = first_token_time - start_time
                    print(f"Time To First Token (TTFT): {ttft:.3f} seconds")
                    assert ttft < 2  # Ensure TTFT is under an acceptable threshold
                    break


# Test Case: Measure Time Per Output Token (TPOT)
@pytest.mark.asyncio
async def test_time_per_output_token():
    start_time = time.time()
    token_times = []

    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            f"{BASE_URL}/chat/sse/",
            json={"session_id": "test_session", "message": "Tell me a story about a fox."}
        ) as response:
            response.raise_for_status()

            async for chunk in response.aiter_text():
                if chunk.strip():
                    token_times.append(time.time())

    tpot = [(token_times[i] - token_times[i - 1]) for i in range(1, len(token_times))]
    avg_tpot = sum(tpot) / len(tpot) if tpot else 0

    print(f"Average Time Per Output Token (TPOT): {avg_tpot:.3f} seconds")
    assert avg_tpot < 0.5  # Ensure TPOT is under an acceptable threshold


# Test Case: Measure Throughput (Tokens per Second)
@pytest.mark.asyncio
async def test_throughput():
    start_time = time.time()
    total_tokens = 0

    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            f"{BASE_URL}/chat/sse/",
            json={"session_id": "test_session", "message": "Explain the theory of relativity in detail."}
        ) as response:
            response.raise_for_status()

            async for chunk in response.aiter_text():
                if chunk.strip():
                    total_tokens += 1

    end_time = time.time()
    duration = end_time - start_time
    throughput = total_tokens / duration if duration > 0 else 0

    print(f"Throughput: {throughput:.3f} tokens/second")
    assert throughput > 5  # Ensure throughput is above an acceptable threshold


# Test Case: Measure Total Response Time
@pytest.mark.asyncio
async def test_total_response_time():
    start_time = time.time()

    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            f"{BASE_URL}/chat/sse/",
            json={"session_id": "test_session", "message": "What is quantum mechanics?"}
        ) as response:
            response.raise_for_status()

            async for chunk in response.aiter_text():
                if chunk.strip():
                    pass  # Consume all tokens

    end_time = time.time()
    total_time = end_time - start_time

    print(f"Total Response Time: {total_time:.3f} seconds")
    assert total_time < 10  # Ensure total response time is under an acceptable threshold
