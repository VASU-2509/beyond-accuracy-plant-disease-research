import type { Metadata } from "next";
import "./globals.css";
import { Shell } from "@/components/layout/shell";
export const metadata: Metadata = { title: "Beyond Accuracy | Plant Disease Research", description: "Domain generalization, calibration and selective prediction research." };
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) { return <html lang="en"><body><Shell>{children}</Shell></body></html>; }
