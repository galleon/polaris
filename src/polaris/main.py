import json

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from smolagents import CodeAgent, HfApiModel

from polaris.tools import IndexerTool, PainterTool

app = FastAPI()
model = HfApiModel("Qwen/Qwen2.5-Coder-32B-Instruct")


@app.post("/v2/chat")
async def chat(
    # "messages" arrives as a string in the form data; we will parse it into a list of dict
    messages: str = Form(...),
    file: UploadFile | None = File(None),
) -> dict:
    try:
        # Parse the JSON string from the multipart form
        messages_list = json.loads(messages)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON in messages form field"}

    # Example: we assume messages_list is a list of dict, each dict has a "content" key
    agent = CodeAgent(tools=[IndexerTool, PainterTool], model=model)

    # Use the last message's content as the prompt
    prompt = messages_list[-1]["content"] if messages_list else ""
    response = agent.run(prompt)

    # Optionally handle file if it’s present
    uploaded_file_name = file.filename if file else None

    return {"response": response, "uploaded_file_name": uploaded_file_name}


if __name__ == "__main__":
    # Launch the application with Uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8888, reload=True)
