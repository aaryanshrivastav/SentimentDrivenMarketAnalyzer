import type { Metadata, Viewport } from 'next'
import { Inter, JetBrains_Mono } from 'next/font/google'
import { Analytics } from '@vercel/analytics/next'
import { GridBackground } from "@/components/grid-background"
import Navbar from '@/components/navbar'
import './globals.css'

const _inter = Inter({ subsets: ['latin'], variable: '--font-inter' })
const _jetbrainsMono = JetBrains_Mono({ subsets: ['latin'], variable: '--font-jetbrains' })

export const metadata: Metadata = {
  title: 'Sentiment Driven Market Analyser',
  description:
    'AI-powered sentiment analysis pipeline for stock market prediction using FinBERT, RoBERTa, XGBoost and LSTM ensemble models.',
}

export const viewport: Viewport = {
  themeColor: '#0a0f0d',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={`${_inter.variable} ${_jetbrainsMono.variable} font-sans antialiased`}>
        
        {/* 🌌 Global Background */}
        <div className="relative min-h-screen bg-background overflow-hidden">
          
          <GridBackground />

          {/* Glow */}
          <div
            className="fixed inset-0 pointer-events-none"
            style={{
              background:
                "radial-gradient(ellipse at 50% 0%, rgba(0,230,118,0.06) 0%, transparent 60%)",
            }}
          />

          {/* Page Content */}
          <div className="relative z-10">
            <Navbar/>
            {children}
          </div>

        </div>

        <Analytics />
      </body>
    </html>
  )
}