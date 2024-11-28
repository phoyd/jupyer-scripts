class MdWriter:
    def __init__(self):
        self.response_text=""
    
    def append(self,delta):
        from IPython.display import display, Markdown, clear_output
        import time
        if delta:  # Add to the response text if available
            self.response_text += delta
            self.show(self.response_text)
            return self

    def show(self,md):
        from IPython.display import display, Markdown, clear_output
        if md:  # Add to the response text if available
            clear_output(wait=True)
            display(Markdown(md))  # Update the output dynamically
            return self

def stats(writer, chunk):
    # model='qwen2.5-coder:1.5b' created_at='2024-11-28T09:32:56.304164926Z' done=True done_reason='stop' total_duration=4470715243 load_duration=40807545 prompt_eval_count=40 prompt_eval_duration=27090000 eval_count=376 eval_duration=4262766000 message=Message(role='assistant', content='', images=None, tool_calls=None)    
    writer.append("\n")
    writer.append("|Key|Value|\n")
    writer.append("|---|-----|\n")
    ns=1/1e9
    writer.append(f"|model|{chunk.model}|\n")
    writer.append(f"|prompt/s|{chunk.prompt_eval_count/(chunk.prompt_eval_duration*ns)}|\n")
    writer.append(f"|reply/s|{chunk.eval_count/(chunk.eval_duration*ns)}|\n")
    print(chunk)
    print(type(chunk))


def ask(prompt, model="qwen2.5-coder:14b", api_host="http://ollama-host:11434"):
    """
    Sends a prompt to a locally hosted OpenAI-compatible backend and streams the response dynamically in a Jupyter Notebook cell.

    Args:
        prompt (str): The user's input prompt.
        model (str): The model to use (default: gpt-3.5-turbo).
        api_host (str): The URL of the locally hosted backend (default: http://ollama-host:11434).
    """
    import requests, ollama
    from IPython.display import display, Markdown, clear_output
    import time,json
    try:
        client = ollama.Client(
              host=api_host,
              headers={'x-some-header': 'some-value'}
            )
        stream = client.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            stream=True,
        )
        writer=MdWriter()
        import spinner
        writer.show(f"![loading]({spinner.circle})")
        for chunk in stream:
            delta=chunk.message.content
            writer.append(delta)            
            if chunk.done: 
                stats(writer,chunk)
                
    except Exception as e:
        print(f"Error: {e}")


from IPython.core.magic import (register_line_magic, register_cell_magic, register_line_cell_magic)

@register_line_magic("?")
def lm_ask(line):
    ask(line)

