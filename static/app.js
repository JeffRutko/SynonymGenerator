(() => {
  const form = document.getElementById("search-form");
  const conceptEl = document.getElementById("concept");
  const contextEl = document.getElementById("context");
  const submitBtn = document.getElementById("submit");
  const progressEl = document.getElementById("progress");
  const resultsEl = document.getElementById("results");

  const EXAMPLES = document.querySelectorAll(".example");

  if (window.marked) {
    marked.setOptions({ gfm: true, breaks: true });
  }

  function renderMarkdown(md) {
    if (!md || !String(md).trim()) {
      return '<p class="muted">Results will appear here…</p>';
    }
    if (window.marked) {
      return marked.parse(md);
    }
    return `<pre>${escapeHtml(md)}</pre>`;
  }

  function renderProgress(md) {
    if (!md || !String(md).trim()) {
      return '<p class="muted">Waiting for a search…</p>';
    }
    if (window.marked) {
      return marked.parse(md);
    }
    return `<pre>${escapeHtml(md)}</pre>`;
  }

  function escapeHtml(text) {
    return String(text)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;");
  }

  EXAMPLES.forEach((btn) => {
    btn.addEventListener("click", () => {
      conceptEl.value = btn.dataset.concept || "";
      contextEl.value = btn.dataset.context || "";
      conceptEl.focus();
    });
  });

  async function readSse(response, onEvent) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let sep;
      while ((sep = buffer.indexOf("\n\n")) >= 0) {
        const chunk = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        for (const line of chunk.split("\n")) {
          if (!line.startsWith("data:")) continue;
          const raw = line.slice(5).trim();
          if (!raw) continue;
          let data;
          try {
            data = JSON.parse(raw);
          } catch {
            continue;
          }
          onEvent(data);
        }
      }
    }
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const concept = conceptEl.value.trim();
    const context = contextEl.value.trim();
    if (!concept) return;

    submitBtn.disabled = true;
    progressEl.innerHTML = renderProgress("- Starting…");
    resultsEl.innerHTML = renderMarkdown("");

    try {
      const response = await fetch("/v1/search", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify({ concept, context }),
      });

      if (!response.ok) {
        const detail = await response.text();
        progressEl.innerHTML = renderProgress(
          `**Error:** ${response.status} ${detail || response.statusText}`
        );
        return;
      }

      await readSse(response, (data) => {
        if (data.done) return;
        if (typeof data.progress === "string") {
          progressEl.innerHTML = renderProgress(data.progress);
        }
        if (typeof data.answer === "string") {
          resultsEl.innerHTML = renderMarkdown(data.answer);
          resultsEl.scrollTop = resultsEl.scrollHeight;
        }
      });
    } catch (err) {
      progressEl.innerHTML = renderProgress(
        `**Failed:** ${err && err.message ? err.message : String(err)}`
      );
    } finally {
      submitBtn.disabled = false;
    }
  });
})();
