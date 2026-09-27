# yuu_v2 UI source

Imported from TsukubaDevGuild/hackathon-2026-autumn-team-b, branch yuu_v2,
commit dc68a64181622876aa9b06cb2f40fb355a25e8fe (2026-09-25 13:45 JST).

- spatial-ui.css: web/my-room-self/spatial-ui.css (unchanged).
- workspace.css: web/my-room-self/style.css (unchanged, renamed).
- Home and room-list HTML use that branch's page structure.
- web/ui-integration.css adapts its layout to the current application controls.

The branch's standalone my-room-self app.js, server.py, video_ai.py and fixed
room models are not used by this integration. Rendering, photo projection,
inference, walking/collisions and memory APIs remain those of app-local.
