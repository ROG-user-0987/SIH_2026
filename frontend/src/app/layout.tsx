import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Voice Clone Detector",
  description: "Real-time voice cloning detection system",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-[#0a0a0f] antialiased">{children}</body>
    </html>
  );
}
