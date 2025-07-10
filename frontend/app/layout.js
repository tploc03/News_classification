import { Inter } from "next/font/google";
import "./globals.css";

// Tải font Inter với bộ ký tự Latin và Tiếng Việt
const inter = Inter({ subsets: ["latin", "vietnamese"] });

export const metadata = {
  title: "News Classification",
  description: "Ứng dụng phân loại tin tức sử dụng Next.js và FastAPI",
};

export default function RootLayout({ children }) {
  return (
    <html lang="vi">
      {/* Áp dụng class của font Inter vào thẻ body */}
      <body className={inter.className}>{children}</body>
    </html>
  );
}
