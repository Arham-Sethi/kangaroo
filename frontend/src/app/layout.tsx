import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Kangaroo Shift",
  description: "Seamless context transfer between LLMs",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
