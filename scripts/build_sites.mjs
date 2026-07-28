import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { spawnSync } from "node:child_process";

const cli = resolve("node_modules", "vinext", "dist", "cli.js");
const result = spawnSync(process.execPath, [cli, "build"], {
  env: process.env,
  stdio: "inherit"
});

if (result.error) {
  throw result.error;
}

const expectedArtifacts = [
  resolve("dist", "server", "index.js"),
  resolve("dist", "client", "index.html")
];
const outputComplete = expectedArtifacts.every((path) => existsSync(path));

if (result.status !== 0 && !(process.platform === "win32" && outputComplete)) {
  process.exit(result.status ?? 1);
}

if (!outputComplete) {
  console.error("Vinext build did not produce the required Sites artifacts.");
  process.exit(1);
}

if (result.status !== 0) {
  console.warn(
    "Vinext completed its Windows static export before a libuv shutdown error; " +
      "verified artifacts will be used."
  );
}

await import("./prepare_sites_output.mjs");
