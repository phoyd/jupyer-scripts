def ask(prompt, model="qwen2.5-coder:7b", api_url="http://ollama-host:11434/v1/chat/completions"):
    """
    Sends a prompt to a locally hosted OpenAI-compatible backend and streams the response dynamically in a Jupyter Notebook cell.

    Args:
        prompt (str): The user's input prompt.
        model (str): The model to use (default: gpt-3.5-turbo).
        api_url (str): The URL of the locally hosted backend (default: http://ollama-host:11434/v1/chat/completions).
    """
    import requests
    from IPython.display import display, Markdown, clear_output
    import time,json
    try:
        # Prepare the request payload
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True
        }
        
        # Send the request to the local backend
        response = requests.post(api_url, json=payload, stream=True)

        # Ensure the request was successful
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}")

        # Initialize an empty string for storing the response
        response_text = ""

        # Process the streamed response
        for line in response.iter_lines(decode_unicode=True):            
            if line.strip():  # Skip empty lines
                if line.startswith("data: "):  # Remove the "data: " prefix
                    line = line[len("data: "):]  # Strip the prefix                
                if line == "[DONE]":  # End of the stream
                    break
                try:
                    # Parse the streamed JSON line
                    chunk = json.loads(line)
                    delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if delta:  # Add to the response text if available
                        response_text += delta
                        clear_output(wait=True)
                        display(Markdown(response_text))  # Update the output dynamically
                        time.sleep(0.02)  # Optional: Slow down updates for readability
                except json.JSONDecodeError:
                    print(f"Could not decode line: {line}")  # Debugging failed chunks
                
    except Exception as e:
        print(f"Error: {e}")
