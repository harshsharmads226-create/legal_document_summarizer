const inputText = document.getElementById("inputText");
const summarizeBtn = document.getElementById("summarizeBtn");
const clearBtn = document.getElementById("clearBtn");
const outputBox = document.getElementById("outputBox");
const errorMsg = document.getElementById("errorMsg");

const statOriginal = document.getElementById("statOriginal");
const statSummary = document.getElementById("statSummary");
const statCompression = document.getElementById("statCompression");

const pdfInput = document.getElementById("pdfInput");
const pdfUploadBtn = document.getElementById("pdfUploadBtn");
const downloadBtn = document.getElementById("downloadBtn");

// Clear
clearBtn.addEventListener("click", () => {
  inputText.value = "";
  outputBox.innerHTML = '<span class="muted">Your summary will appear here…</span>';
  errorMsg.textContent = "";
  statOriginal.textContent = "Original: 0 sentences";
  statSummary.textContent = "Summary: 0 sentences";
  statCompression.textContent = "Compression: 0%";
});

// Summarize text
summarizeBtn.addEventListener("click", async () => {
  const text = inputText.value.trim();
  errorMsg.textContent = "";

  if (!text) {
    errorMsg.textContent = "Please enter text to summarize.";
    return;
  }

  try {
    const res = await fetch("/summarize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    const data = await res.json();
    if (!data.success) {
      outputBox.textContent = "Error: " + data.error;
      return;
    }

    updateSummaryUI(data);
  } catch {
    outputBox.textContent = "Request failed. Check backend.";
  }
});

// PDF upload
pdfUploadBtn.addEventListener("click", async () => {
  const file = pdfInput.files[0];
  errorMsg.textContent = "";

  if (!file) {
    errorMsg.textContent = "Please select a PDF.";
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/upload-pdf", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    if (!data.success) {
      outputBox.textContent = "Error: " + data.error;
      return;
    }

    inputText.value = data.extracted_text;
    updateSummaryUI(data);
  } catch {
    outputBox.textContent = "PDF upload failed.";
  }
});

// UI update
function updateSummaryUI(data) {
  outputBox.textContent = data.summary;
  statOriginal.textContent = `Original: ${data.total_sentences} sentences`;
  statSummary.textContent = `Summary: ${data.summary_sentences} sentences`;

  const comp = data.total_sentences
    ? Math.round(100 - (data.summary_sentences / data.total_sentences) * 100)
    : 0;

  statCompression.textContent = `Compression: ${comp}%`;
}

// Download
downloadBtn.addEventListener("click", () => {
  const text = outputBox.textContent.trim();
  if (!text) return alert("Nothing to download.");

  const blob = new Blob([text], { type: "text/plain" });
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = "legal_summary.txt";
  a.click();

  URL.revokeObjectURL(url);
});
