const form = document.getElementById("upload-form");
const audioInput = document.getElementById("audio-file");
const analyzeBtn = document.getElementById("analyze-btn");

const audioPlayer = document.getElementById("audio-player");
const resultsDiv = document.getElementById("results");
const karaokeText = document.getElementById("karaoke-text");
const jumpBtn = document.getElementById("jump-btn");
const segmentsList = document.getElementById("segments-list");

let karaokeStart = null;

function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = (seconds % 60).toFixed(1).padStart(4, "0");
  return `${m}:${s}`;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const file = audioInput.files[0];
  if (!file) {
    karaokeText.textContent = "Please select an audio file.";
    resultsDiv.style.display = "block";
    return;
  }

  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing…";
  karaokeText.textContent = "Analyzing audio, please wait…";
  segmentsList.innerHTML = "";
  karaokeStart = null;
  resultsDiv.style.display = "block";

  try {
    audioPlayer.src = URL.createObjectURL(file);
    audioPlayer.load();

    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("/upload", {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Prediction failed");
    }

    karaokeStart = data.karaoke_start;

    if (karaokeStart === null) {
      karaokeText.textContent = "No karaoke start detected.";
    } else {
      karaokeText.textContent = `Suggested karaoke start: ${formatTime(karaokeStart)}`;
    }

    segmentsList.innerHTML = "";

    if (!data.segments || data.segments.length === 0) {
      segmentsList.innerHTML = "<li>No lyric segments detected.</li>";
    } else {
      data.segments.forEach(seg => {
        const li = document.createElement("li");
        li.textContent = `Lyrics from ${formatTime(seg.start)} → ${formatTime(seg.end)}`;
        segmentsList.appendChild(li);
      });
    }

  } catch (err) {
    karaokeText.textContent = err.message;
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyze";
  }
});

jumpBtn.addEventListener("click", () => {
  if (karaokeStart !== null) {
    audioPlayer.currentTime = karaokeStart;
    audioPlayer.play();
  }
});
