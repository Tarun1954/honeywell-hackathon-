import { copyFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";

const source = resolve(".openai", "hosting.json");
const destination = resolve("dist", ".openai", "hosting.json");

mkdirSync(dirname(destination), { recursive: true });
copyFileSync(source, destination);

console.log("Prepared Sites metadata in dist/.openai/hosting.json");
