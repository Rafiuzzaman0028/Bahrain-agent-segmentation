# app.py
import os
import logging
import gradio as gr

from bahrain_agent.agent import BahrainStatsAgent
from bahrain_agent.nlu_router import route_and_answer  # optional LLM-refinement wrapper

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DATA_PATH = os.getenv("BAHRAIN_DATA_PATH", "data/bahrain_master")
agent = BahrainStatsAgent(data_path=DATA_PATH)
LOG.info("Loading BahrainStatsAgent with data path: %s", DATA_PATH)

# Whether to call LLM refinement (controlled by checkbox in UI)
DEFAULT_USE_LLM = True

def submit_message(message: str, history: list, use_llm: bool):
    """
    message: str input from user textbox
    history: current list of messages in {'role','content'} format
    use_llm: boolean whether to refine answer with LLM
    Returns: updated history for the Chatbot, and empty string for the textbox
    """
    if not message or not message.strip():
        return history or [], ""

    history = history or []

    # append user message (messages format)
    user_msg = {"role": "user", "content": message.strip()}
    history.append(user_msg)

    # produce answer via agent, optionally refine with LLM via route_and_answer
    try:
        if use_llm:
            answer = route_and_answer(agent, message.strip(), use_llm=True)
        else:
            answer = agent.answer_question(message.strip())
    except Exception as e:
        LOG.exception("Error calling agent:")
        answer = f"Error producing answer: {e}"

    assistant_msg = {"role": "assistant", "content": answer}
    history.append(assistant_msg)

    # return new history to Chatbot (messages format) and clear textbox
    return history, ""

def clear_history():
    return []

with gr.Blocks(title="BH Bahrain Statistical AI Agent") as demo:
    gr.Markdown("## BH Bahrain Statistical AI Agent\nAsk about labour, households, population density, housing, segmentation etc.")
    with gr.Row():
        llm_checkbox = gr.Checkbox(value=DEFAULT_USE_LLM, label="Use LLM (ChatGPT) refinement")
    chatbot = gr.Chatbot(label="Chat", elem_id="chatbot")  # modern Gradio will accept messages format
    txt = gr.Textbox(placeholder="Type your question...", show_label=False)
    with gr.Row():
        send_btn = gr.Button("Send")
        clear_btn = gr.Button("Clear")

    state = gr.State(value=[])  # list of {'role','content'} messages

    # wire up events
    send_btn.click(fn=submit_message, inputs=[txt, state, llm_checkbox], outputs=[chatbot, txt], queue=True).then(
        lambda h: h, inputs=[chatbot], outputs=[state]
    )
    txt.submit(fn=submit_message, inputs=[txt, state, llm_checkbox], outputs=[chatbot, txt], queue=True).then(
        lambda h: h, inputs=[chatbot], outputs=[state]
    )
    clear_btn.click(fn=clear_history, inputs=None, outputs=[chatbot, state])

if __name__ == "__main__":
    # choose any free port if desired; change server_port or set GRADIO_SERVER_PORT in env.
    demo.launch(server_name="127.0.0.1", server_port=int(os.getenv("GRADIO_PORT", 7860)), share=False)
