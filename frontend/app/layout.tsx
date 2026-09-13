import type { Metadata } from "next";
import { Geist, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { NavBar } from "./_components/NavBar";

const geist = Geist({
  variable: "--font-geist",
  subsets: ["latin"],
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-jetbrains-mono",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "FinSight Alpha",
  description:
    "A real-time analytical execution engine for SEC filings: hybrid retrieval, multi-hop reasoning, and citation-grounded synthesis, with every figure marked by the rigor behind it.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geist.variable} ${jetbrainsMono.variable} h-full`}
    >
      <body className="min-h-full flex flex-col font-sans antialiased">
        <div className="grid-texture pointer-events-none fixed inset-x-0 top-0 h-[480px]" />
        <NavBar />
        <main className="relative flex-1">{children}</main>
      </body>
    </html>
  );
}
