import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RL Taxi Dispatch — Live Dashboard",
  description: "Real-time multi-agent taxi fleet dispatch visualisation",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-[#0a0a0f] antialiased">{children}</body>
    </html>
  );
}
