import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Federal Student Loan Assistant',
  description: 'AI-powered customer service assistant for federal student loan questions',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}