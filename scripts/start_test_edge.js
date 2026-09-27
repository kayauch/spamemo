const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");

const project = path.resolve(__dirname, "..");
const scene = process.argv[2] || "relocalization-smoke";
const port = Number(process.argv[3] || process.env.SPATIAL_DEMO_PORT || 18001);
const candidates = [
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
];
const edge = candidates.find(fs.existsSync);
if (!edge) throw new Error("Microsoft Edge not found");

const fakeCamera = path.join(project, "test-assets", "fake-camera.y4m");
if (!fs.existsSync(fakeCamera)) throw new Error(`Missing fake camera: ${fakeCamera}`);
const profile = path.join(project, "test-assets", "team-smoke-edge-profile");
fs.mkdirSync(profile, { recursive: true });
const url = `http://127.0.0.1:${port}/?scene=${encodeURIComponent(scene)}`;

const args = [
  "--headless=new",
  "--remote-debugging-port=9222",
  `--user-data-dir=${profile}`,
  "--use-fake-ui-for-media-stream",
  "--use-fake-device-for-media-stream",
  `--use-file-for-fake-video-capture=${fakeCamera}`,
  "--autoplay-policy=no-user-gesture-required",
  url,
];
const child = spawn(edge, args, {
  detached: true,
  stdio: "ignore",
  windowsHide: true,
});
child.unref();
console.log(child.pid);
