import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";
import { REPO_URL } from "@/lib/api";

const inter = Inter({ variable: "--font-inter", subsets: ["latin"] });
const mono = JetBrains_Mono({ variable: "--font-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "NearZeroFP Vision: agentic false-positive triage",
  description:
    "Detector flags only what it's unsure about; a VLM agent verifies, extracts attributes, escalates the rest, and the decisions train a cheaper open model.",
};

const NAV = [
  { href: "/console", label: "Console" },
  { href: "/review", label: "Review queue" },
  { href: "/search", label: "Search" },
  { href: "/loop", label: "Loop & distillation" },
];

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className={`${inter.variable} ${mono.variable} min-h-screen`}>
        <header className="sticky top-0 z-30 border-b border-line bg-bg/85 backdrop-blur">
          <div className="mx-auto flex h-14 max-w-[1400px] items-center gap-6 px-4">
            <Link href="/" className="flex items-center gap-2 font-semibold tracking-tight">
              <span className="relative inline-flex h-3 w-3">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-accent opacity-40" />
                <span className="relative inline-flex h-3 w-3 rounded-full bg-accent" />
              </span>
              NearZeroFP Vision
            </Link>
            <nav className="flex flex-1 items-center gap-1 overflow-x-auto text-sm text-muted">
              {NAV.map((n) => (
                <Link key={n.href} href={n.href} className="whitespace-nowrap rounded px-3 py-1.5 hover:bg-panel-2 hover:text-text">
                  {n.label}
                </Link>
              ))}
            </nav>
            <a href={REPO_URL} target="_blank" rel="noreferrer" className="hidden text-sm text-muted hover:text-text sm:block">
              GitHub ↗
            </a>
          </div>
        </header>
        <main>{children}</main>
        <footer className="mt-16 border-t border-line">
          <div className="mx-auto max-w-[1400px] px-4 py-6 text-xs text-muted">
            Independent portfolio demo, not affiliated with any vendor. Clips: Intel IoT DevKit sample-videos (CC BY 4.0); night/rain and dusk variants
            are synthetic degradations. Detector: Ultralytics YOLO11n (AGPL-3.0). Verifier: Claude (Anthropic). Student: Qwen2.5-VL + LoRA.
          </div>
        </footer>
      </body>
    </html>
  );
}
