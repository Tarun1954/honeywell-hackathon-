import { createReadStream, existsSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { extname, resolve, sep } from "node:path";

const clientRoot = resolve(import.meta.dirname, "..", "client");
const port = Number.parseInt(process.env.PORT ?? "3000", 10);
const host = process.env.HOST ?? "0.0.0.0";

const contentTypes = new Map([
  [".css", "text/css; charset=utf-8"],
  [".csv", "text/csv; charset=utf-8"],
  [".html", "text/html; charset=utf-8"],
  [".ico", "image/x-icon"],
  [".js", "text/javascript; charset=utf-8"],
  [".json", "application/json; charset=utf-8"],
  [".png", "image/png"],
  [".svg", "image/svg+xml"],
  [".txt", "text/plain; charset=utf-8"],
  [".webp", "image/webp"],
  [".woff", "font/woff"],
  [".woff2", "font/woff2"]
]);

function sendHeaders(response, status, contentType, contentLength) {
  response.writeHead(status, {
    "Cache-Control": "public, max-age=0, must-revalidate",
    "Content-Length": contentLength,
    "Content-Security-Policy":
      "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; " +
      "script-src 'self' 'unsafe-inline'; connect-src 'self'",
    "Content-Type": contentType,
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY"
  });
}

function approvedPath(pathname) {
  const relativePath = pathname === "/" ? "index.html" : pathname.slice(1);
  const candidate = resolve(clientRoot, relativePath);
  if (candidate !== clientRoot && !candidate.startsWith(`${clientRoot}${sep}`)) {
    return null;
  }
  return candidate;
}

const server = createServer((request, response) => {
  if (!["GET", "HEAD"].includes(request.method ?? "")) {
    const message = Buffer.from("Method Not Allowed");
    sendHeaders(
      response,
      405,
      "text/plain; charset=utf-8",
      message.byteLength
    );
    response.end(request.method === "HEAD" ? undefined : message);
    return;
  }

  let pathname;
  try {
    pathname = decodeURIComponent(
      new URL(request.url ?? "/", "http://localhost").pathname
    );
  } catch {
    const message = Buffer.from("Bad Request");
    sendHeaders(response, 400, "text/plain; charset=utf-8", message.byteLength);
    response.end(request.method === "HEAD" ? undefined : message);
    return;
  }

  let filePath = approvedPath(pathname);
  if (
    filePath &&
    existsSync(filePath) &&
    statSync(filePath).isDirectory()
  ) {
    filePath = resolve(filePath, "index.html");
  }

  if (!filePath || !existsSync(filePath) || !statSync(filePath).isFile()) {
    filePath = resolve(clientRoot, "404.html");
  }

  const status = filePath.endsWith(`${sep}404.html`) ? 404 : 200;
  const size = statSync(filePath).size;
  const contentType =
    contentTypes.get(extname(filePath).toLowerCase()) ??
    "application/octet-stream";
  sendHeaders(response, status, contentType, size);

  if (request.method === "HEAD") {
    response.end();
    return;
  }
  createReadStream(filePath).pipe(response);
});

server.listen(port, host, () => {
  console.log(`Eco-Loop dashboard listening on http://${host}:${port}`);
});

function shutdown() {
  server.close(() => process.exit(0));
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
