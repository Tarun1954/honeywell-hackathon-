import { copyFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";

const source = resolve(".openai", "hosting.json");
const destination = resolve("dist", ".openai", "hosting.json");
const serverSource = resolve("scripts", "sites_static_server.mjs");
const serverDestination = resolve("dist", "server", "index.js");

mkdirSync(dirname(destination), { recursive: true });
copyFileSync(source, destination);
copyFileSync(serverSource, serverDestination);

console.log("Prepared Sites metadata and static server entrypoint");
