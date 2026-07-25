# Demo recording checklist

## Before recording

- [ ] Close email, chat, password managers, notifications, and unrelated
  browser tabs.
- [ ] Do not open or print `.env`. Confirm the terminal history does not show
  credentials, personal paths, tokens, or private repository URLs.
- [ ] Open these windows in advance:
  - `docs/system_architecture.md` at the Mermaid architecture diagram
  - `docs/dashboard/index.html` in a browser
  - a PowerShell terminal at the repository root
  - `docs/demo_script.md` on a second screen or non-recorded device
- [ ] In the dashboard, preload the energy, comfort, and action/latency charts.
- [ ] Set terminal font size to at least 18 px and browser zoom to 110-125%.
- [ ] Hide desktop icons, bookmarks, account avatars, usernames, and the
  Windows taskbar if it contains personal information.
- [ ] Enable Do Not Disturb.

## Choose the command

Normal live-first command with automatic saved-evidence fallback:

```powershell
python -m scripts.run_final_demo --mode auto
```

Reliability-first command when Ollama, the local model, or EnergyPlus cannot be
used during recording:

```powershell
python -m scripts.run_final_demo --mode replay
```

If replay is used, keep the replay banner visible and say that it is previously
captured verified evidence. Never call it a live run.

## Screen layout and narration order

- [ ] Record at 1080p or higher.
- [ ] Use one main window at a time; avoid rapid switching or tiny split panes.
- [ ] Start on the dashboard headline.
- [ ] Show the architecture diagram.
- [ ] Switch to the terminal and run the launcher.
- [ ] Follow launcher sections 1-8:
  architecture, sensors, MCP, Ollama action, validation, writeback,
  EnergyPlus status, results.
- [ ] Finish on the dashboard charts and result headline.
- [ ] Follow the exact timed order in `docs/demo_script.md`.

## Audio and timing

- [ ] Select the intended microphone and disable noisy automatic input
  switching.
- [ ] Record a ten-second test; confirm voice clarity, no clipping, and no fan
  noise.
- [ ] Start a three-minute timer when narration begins.
- [ ] Speak at approximately 125-135 words per minute.
- [ ] Do not troubleshoot during the take. If the live path stalls, use replay
  mode for the next take.
- [ ] Stop by 2:55, leaving a small editing margin under three minutes.

## After recording

- [ ] Save the eventual final file as
  `submission/video/eco_loop_demo.mp4`.
- [ ] Play the exported video from beginning to end with audio enabled.
- [ ] Confirm duration is no more than 3:00.
- [ ] Confirm text is readable and the replay/live banner is visible.
- [ ] Confirm no credentials, `.env` contents, personal notifications, or
  hidden chain-of-thought appear.
- [ ] Confirm the final results and comfort trade-off are stated accurately.
- [ ] Do not create the final submission ZIP until the video and presentation
  are both complete.
