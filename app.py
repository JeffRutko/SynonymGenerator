import markdown
import gradio as gr

from synonym_agent import generate_synonyms_stream


EXAMPLE_CONCEPT = "the base station transmits a packet to a UE"
EXAMPLE_CONTEXT = "H04W CPC Classification area"

# Fixed-pixel inner scroller — HF Spaces iframes often won't grow, so page
# scroll fails. Do not use vh/% heights (breaks fullscreen + iframe resize).
RESULTS_CSS = """
#results-scroll {
  height: 520px;
  overflow-y: scroll;
  overflow-x: auto;
  padding: 16px 16px 96px 16px;
  box-sizing: border-box;
  border: 1px solid rgba(128, 128, 128, 0.4);
  border-radius: 8px;
  text-align: left;
}
#results-scroll table {
  border-collapse: collapse;
  width: 100%;
  margin: 0.75rem 0;
}
#results-scroll th,
#results-scroll td {
  border: 1px solid rgba(128, 128, 128, 0.35);
  padding: 0.35rem 0.55rem;
}
#results-scroll pre {
  overflow-x: auto;
  padding: 0.75rem;
  border-radius: 6px;
  background: rgba(127, 127, 127, 0.12);
}
"""


def render_results_html(answer_md: str) -> str:
    if not answer_md.strip():
        body = "<p><em>Results will appear here…</em></p>"
    else:
        body = markdown.markdown(
            answer_md,
            extensions=["tables", "fenced_code", "sane_lists", "nl2br"],
        )
    # Inline styles so HF Spaces / Gradio scoping cannot drop the scrollbar.
    return f"""
<div id="results-scroll" style="
  height: 520px;
  overflow-y: scroll;
  overflow-x: auto;
  padding: 16px 16px 96px 16px;
  box-sizing: border-box;
  border: 1px solid rgba(128, 128, 128, 0.4);
  border-radius: 8px;
  text-align: left;
">{body}</div>
"""


async def run_synonym_search(concept: str, context: str):
    async for progress, answer in generate_synonyms_stream(
        concept or "", context or ""
    ):
        yield progress, render_results_html(answer)


with gr.Blocks(title="Conceptual Search Helper") as demo:
    gr.Markdown(
        """
        # Conceptual Search Helper

        Enter a technical concept or phrase. The agent uses patent and telecom
        search tools plus retrieval to propose synonyms, **CPC subgroups**, and
        Boolean search strings you can reuse in literature or patent databases.
        """
    )

    with gr.Row():
        concept = gr.Textbox(
            label="Concept / phrase",
            placeholder=EXAMPLE_CONCEPT,
            lines=3,
            scale=3,
        )
        context = gr.Textbox(
            label="Optional domain / CPC hint",
            placeholder=EXAMPLE_CONTEXT,
            lines=3,
            scale=2,
        )

    gr.Examples(
        examples=[
            [EXAMPLE_CONCEPT, EXAMPLE_CONTEXT],
            [
                "coordinated multipoint handoff in cellular networks",
                "H04W CPC Classification area",
            ],
        ],
        inputs=[concept, context],
    )

    submit = gr.Button("Generate search help", variant="primary")
    progress = gr.Markdown(label="Progress", container=True)
    results = gr.HTML(label="Results", container=True)

    submit.click(
        fn=run_synonym_search,
        inputs=[concept, context],
        outputs=[progress, results],
    )


if __name__ == "__main__":
    demo.launch(css=RESULTS_CSS)
