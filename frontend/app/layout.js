import "./globals.css";

export const metadata = {
  title: "LeafLens — AI Crop Disease Diagnosis",
  description:
    "Photograph a crop leaf and get an instant disease diagnosis with grounded treatment advice in Bangla and English. Built for Bangladeshi farmers.",
};

export const viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#1b7a43",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
