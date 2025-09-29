import os
import sys
import time

import litellm

# list of providers per model via openrouter
# note that each model can have multiple providers, whilst they might
# state their quantisation it doesn't mean two fp8s are the same!
# I have restricted providers to 1 to simplify everything
providers = {
    # DeepSeek V3 0324
    # https://openrouter.ai/deepseek/deepseek-chat-v3-0324
    "openrouter/deepseek/deepseek-chat-v3-0324": {
        "order": [
            "siliconflow/fp8",  # SiliconFlow, fp8 164k 164k # WORKING GOOD
            # "nebius/fp8",  # "Nebius AI Studio",  # fp8 164k, 164k WORKING GOOD bit slower
            # "lambda/fp8",  # "Lambda",  # fp8, 164k, 164k DEAD
            # "beseten/fp8", # Beseten, fp8 164k context 131k length DEAD
            # "gmicloud/fp8", # GMiCloud, fp8 164k 164k UNRESP
            # "chutes/fp8", # Chutes, fp8 164k 164k # did try
            # "deepinfra/fp8",  # "DeepInfra",  # fp8, 164k, 164k DEAD
        ],
        "allow_fallbacks": False,
    },
    # Opus 4 is multimodal (text and image)
    "openrouter/anthropic/claude-opus-4": {
        "order": [
            "anthropic",  # 200k, 32k
            "google-vertex",  # 200k, 32k
        ],
        "allow_fallbacks": False,
    },
    # GPT 4o, multimodal
    # https://openrouter.ai/openai/gpt-4o/api
    # GPT o3 - needs private (2nd) OpenAI key so I'm ignoring it
    # GPT o1-pro is crazy expensive! Multimodal
    # https://openrouter.ai/openai/o1-pro
    # GPT o1 is more like Opus 4 on pricing, multimodal
    # https://openrouter.ai/openai/o1
}


def ensure_system_certs():
    """Ensure Python SSL uses system certificates.

    Tries pip-system-certs' truststore injection first; silently
    continues if unavailable. Must be called before any network I/O.
    """
    try:
        # Prefer pip-system-certs which integrates with pip's vendored truststore
        import pip_system_certs.wrapt_requests as _psc  # type: ignore

        _psc.inject_truststore()
        return
    except Exception:
        # As a fallback, try the public truststore package if present
        try:
            import truststore  # type: ignore

            truststore.inject_into_ssl()
        except Exception:
            # Leave default cert handling in place
            pass


def disable_litellm_logging():
    # disable litllm debug logs
    # https://github.com/BerriAI/litellm/issues/6813
    litellm._logging._disable_debugging()


def check_litellm_key(args):
    # Ensure SSL uses system trust store before validating key (prevents TLS/auth false negatives)
    ensure_system_certs()
    api_key_valid = litellm.check_valid_key(
        api_key=os.environ["OPENROUTER_API_KEY"], model=args.model_name
    )
    assert api_key_valid is True, "API key is not valid"


def call_llm(model, messages):
    """Call llm, get a response (hopefully)

    Returns:
        response: litellm.Response object, content"""
    provider = providers[model]
    NBR_PROVIDER_RETRIES = 10
    response = None
    # check we have a well-formed message to send
    for message in messages:
        assert "content" in message.keys()
        assert "role" in message.keys()
    for retry_n in range(NBR_PROVIDER_RETRIES):
        temperature = 1.0
        if temperature != 1.0:
            print(f"~~~~~~~~~~~~~# non default TEMPERATURE {temperature}")
        try:
            # https://openrouter.ai/docs/features/provider-routing#allowing-only-specific-providers
            response = litellm.completion(
                model=model,
                messages=messages,
                transforms=[""],
                route="",
                provider=provider,
                timeout=20,
                temperature=temperature,
            )
        except litellm.RateLimitError as e:
            print(f"!Rate limit error: {e}")
            print("Sleeping for a bit...")
            time.sleep(2)
        except litellm.Timeout as e:
            print(f"!Timeout error: {e}")
            print("Sleeping for a bit...")
            time.sleep(2)
        except litellm.APIError as e:
            print(f"!APIError error: {e}")
            print("Sleeping for a bit...")
            time.sleep(2)
        if response is not None:
            break
    if response is None:
        print("Failed to get a response after retries")
        sys.exit(1)
    return response, response.choices[0].message.content
