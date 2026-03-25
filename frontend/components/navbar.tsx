'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'

const navItems = [
  { name: 'Demo', href: '/demo' },
  { name: 'Company', href: '/company' },
  { name: 'Tweet', href: '/tweet' },
]

export default function Navbar() {
  const pathname = usePathname()

  // Determine indicator position
  const activeIndex = navItems.findIndex(item => item.href === pathname)

  return (
    <div className="sticky top-0 z-50 flex justify-center pt-6">
      <div className="w-[92%] max-w-5xl">

        {/* 🔹 Outer Shell */}
        <div className="relative rounded-2xl border border-white/10 
          bg-black/30 backdrop-blur-xl px-5 py-3
          shadow-[0_20px_60px_rgba(0,0,0,0.6)]">

          {/* top light line */}
          <div className="absolute top-0 left-0 right-0 h-[1px]
            bg-gradient-to-r from-transparent via-white/20 to-transparent" />

          <div className="flex items-center justify-between">

            {/* 🔸 Logo */}
            <div className="text-white/80 font-medium tracking-wide flex items-center gap-2">
              <span className="text-[#00e676]">⬢</span>
              <span className="hidden sm:block">Sentiment Driven Market Analyzer</span>
            </div>

            {/* 🔸 NAV TRACK */}
            <div className="relative flex bg-black/40 p-1 rounded-xl border border-white/5 overflow-hidden">

              {/* ⚡ Sliding Indicator */}
              <div
                className="absolute top-1 bottom-1 w-[calc(100%/3-6px)] rounded-lg
                  bg-[#00e676]/10
                  shadow-[0_0_25px_rgba(0,230,118,0.35)]
                  transition-all duration-500 ease-[cubic-bezier(0.22,1,0.36,1)]"
                style={{
                  transform: `translateX(${activeIndex * 100}%)`,
                }}
              />

              {navItems.map((item) => {
                const isActive = pathname === item.href

                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className="relative z-10 px-5 py-2 text-sm font-medium text-center w-[120px]"
                  >
                    <span
                      className={`transition-all duration-300
                        ${isActive
                          ? 'text-[#00e676] scale-105'
                          : 'text-white/60 hover:text-white'}`}
                    >
                      {item.name}
                    </span>
                  </Link>
                )
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}