from openai import OpenAI

def nemotron():
    client = OpenAI(
    base_url = "https://integrate.api.nvidia.com/v1",
    api_key = "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"
    )


    completion = client.chat.completions.create(
    model="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
    messages=[{"role":"user","content":"How can I play basketball?"}],
    temperature=1.00,
    top_p=0.01,
    max_tokens=1024,
    stream=True
    )


    lines = ""

    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            text = chunk.choices[0].delta.content
            lines += text

    # Convert literal "\n" to actual newlines
    lines = lines.replace("\\n", "\n")
    
    return lines
