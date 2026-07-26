import "./globals.css";

export const metadata = {
  title: "Eco-Loop | Verified Building-Agent Results",
  description:
    "Verified EnergyPlus, MCP, and Ollama building-control results with safety evidence and an explicit comfort trade-off."
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
