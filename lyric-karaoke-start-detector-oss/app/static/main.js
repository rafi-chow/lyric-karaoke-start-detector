const form = document.getElementById("upload-form");
const audioInput = document.getElementById("audio-file");
const audioPlayer = document.getElementById("audio-player");
const resultsDiv = document.getElementById("results");
const karaokeText = document.getElementById("karaoke-text");
const jumpBtn = document.getElementById("jump-btn");
const segmentsList = document.getElementById("segments-list");

let karaokeStart = null;

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const file = audioInput.files[0];
  if (!file) return;


  audioPlayer.src = URL.createObjectURL(file);
  resultsDiv.style.display = "block";

  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("/upload", {
    method: "POST",
    body: formData
  });

  const data = await response.json();

  segmentsList.innerHTML = "";

  karaokeStart = data.karaoke_start;

  if (karaokeStart === null) {
    karaokeText.textContent = "No karaoke start detected.";
  } else {
    karaokeText.textContent =
      `Suggested karaoke start: ${karaokeStart.toFixed(1)} seconds`;
  }

  data.segments.forEach(seg => {
    const li = document.createElement("li");
    li.textContent =
      `Lyrics from ${seg.start.toFixed(1)}s to ${seg.end.toFixed(1)}s`;
    segmentsList.appendChild(li);
  });
});

jumpBtn.addEventListener("click", () => {
  if (karaokeStart !== null) {
    audioPlayer.currentTime = karaokeStart;
    audioPlayer.play();
  }
});
