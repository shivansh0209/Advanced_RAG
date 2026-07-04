import { useState, FormEvent, useEffect } from "react";
import { 
  Scale, 
  Search, 
  User, 
  ArrowUp, 
  ArrowLeft, 
  Sparkles, 
  Loader2, 
  RotateCcw,
  CheckCircle2,
  FileText,
  AlertCircle
} from "lucide-react";
import { motion, AnimatePresence } from "motion/react";
import { MOCK_RESPONSES, DEFAULT_RESPONSE, LegalResponse } from "./mockData";

// List of the 7 Indian Acts
const INDIAN_ACTS = [
  { id: "CrPC", name: "Code of Criminal Procedure" },
  { id: "RTI", name: "Right to Information Act" },
  { id: "HMA", name: "Hindu Marriage Act" },
  { id: "IT Act", name: "Information Technology Act" },
  { id: "CPA", name: "Consumer Protection Act" },
  { id: "ICA", name: "Indian Contract Act" },
  { id: "IPC", name: "Indian Penal Code" }
];

// Recommended question chips
const RECOMMENDED_CHIPS = [
  { text: "What are the grounds for divorce in HMA?", act: "HMA" },
  { text: "Penalty for theft under IPC?", act: "IPC" },
  { text: "RTI application process", act: "RTI" }
];

export default function App() {
  const [screenState, setScreenState] = useState<"EMPTY" | "LOADING" | "ANSWER">("EMPTY");
  const [searchQuery, setSearchQuery] = useState("");
  const [activeAct, setActiveAct] = useState<string | null>(null);
  const [currentResponse, setCurrentResponse] = useState<LegalResponse | null>(null);
  const [submittedQuery, setSubmittedQuery] = useState("");

  // Handle click on top Act pill
  const handleActClick = (actId: string) => {
    setActiveAct(actId === activeAct ? null : actId);
    
    // Auto-fill related question based on act clicked
    if (actId === "HMA") {
      setSearchQuery("What are the grounds for divorce in HMA?");
    } else if (actId === "IPC") {
      setSearchQuery("Penalty for theft under IPC?");
    } else if (actId === "RTI") {
      setSearchQuery("RTI application process");
    } else if (actId === "ICA") {
      setSearchQuery("What are the legal implications of a breach of contract under the Indian Contract Act for service-level agreements?");
    } else {
      // Generic question template for other acts
      setSearchQuery(`What are the key features of the ${actId}?`);
    }
  };

  // Handle recommendation chip click
  const handleChipClick = (text: string, act: string) => {
    setSearchQuery(text);
    setActiveAct(act);
  };

  // Handle form submission
  const onSubmit = async (e?: FormEvent) => {
    if (e) e.preventDefault();
    if (!searchQuery.trim()) return;

    const query = searchQuery.trim();
    setSearchQuery("");
    setSubmittedQuery(query);
    setScreenState("LOADING");

    try {
      const res = await fetch('http://localhost:8000/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: query })
      });

      const data = await res.json();

      setCurrentResponse({
        question: query,
        answer: data.answer,
        retrievedFrom: data.acts_cited   // ← matches LegalResponse type
      });
      setScreenState("ANSWER");

    } catch (error) {
      console.error("API error:", error);
      setCurrentResponse(DEFAULT_RESPONSE("API not responding"));
      setScreenState("ANSWER");
    }
  };

  // Reset to empty home state
  const handleReset = () => {
    setSearchQuery("");
    setSubmittedQuery("");
    setActiveAct(null);
    setCurrentResponse(null);
    setScreenState("EMPTY");
  };

  // Parse response text to render any "Section XX" or "Section XXA" as inline green monospace pills
  const parseAnswerText = (text: string) => {
    // Regex matches "Section" followed by space and one or more digits, optionally followed by uppercase letters
    const regex = /\b(Section\s+\d+[A-Z]?)\b/g;
    const parts = text.split(regex);

    return parts.map((part, index) => {
      if (part.match(/\bSection\s+\d+[A-Z]?\b/)) {
        return (
          <span
            key={index}
            id={`section-pill-${index}`}
            className="inline-flex items-center px-2 py-0.5 rounded bg-emerald-950/60 text-emerald-400 border border-emerald-800/40 font-mono text-xs font-medium mx-1 transition-all"
          >
            {part}
          </span>
        );
      }
      return <span key={index}>{part}</span>;
    });
  };

  return (
    <div className="min-h-screen bg-black text-zinc-100 flex flex-col font-sans selection:bg-emerald-900/40 selection:text-emerald-300">
      
      {/* Dynamic Header */}
      <header className="border-b border-zinc-900 bg-black/90 sticky top-0 z-50 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          
          {/* Logo Brand */}
          <div 
            onClick={handleReset}
            className="flex items-center space-x-3 cursor-pointer group"
          >
            <div className="p-2 bg-zinc-900 rounded-lg border border-zinc-800 group-hover:border-zinc-700 transition-colors">
              <Scale className="h-5 w-5 text-emerald-500" />
            </div>
            <div className="flex items-baseline space-x-2">
              <span className="font-bold text-lg tracking-tight text-white">LexAI</span>
              <span className="hidden sm:inline-block h-4 w-[1px] bg-zinc-800"></span>
              <span className="hidden sm:inline-block text-xs font-mono tracking-widest text-zinc-500 uppercase">
                Indian Legal Intelligence
              </span>
            </div>
          </div>

          {/* Quick Act Filters */}
          {/* <div className="hidden md:flex items-center space-x-1.5">
            {INDIAN_ACTS.map((act) => {
              const isSelected = activeAct === act.id;
              return (
                <button
                  key={act.id}
                  onClick={() => handleActClick(act.id)}
                  title={act.name}
                  className={`px-3 py-1 text-xs font-medium rounded-full transition-all duration-200 border ${
                    isSelected
                      ? "bg-emerald-950/80 text-emerald-400 border-emerald-800/60"
                      : "bg-zinc-900/60 text-zinc-400 border-zinc-800 hover:text-zinc-200 hover:border-zinc-700"
                  }`}
                >
                  {act.id}
                </button>
              );
            })}
          </div> */}

          {/* User & Search Controls */}
          <div className="flex items-center space-x-4">
            <button 
              onClick={() => {
                if (screenState !== "EMPTY") handleReset();
              }}
              className="p-2 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900/60 rounded-lg transition-all"
              title="New Search"
            >
              <Search className="h-5 w-5" />
            </button>
            <div className="h-8 w-8 rounded-full bg-zinc-900 border border-zinc-800 flex items-center justify-center cursor-pointer hover:border-zinc-700 transition-all">
              <User className="h-4 w-4 text-zinc-400" />
            </div>
          </div>

        </div>

        {/* Small screen Act Filters bar */}
        <div className="md:hidden flex items-center space-x-1.5 overflow-x-auto px-4 py-2 border-t border-zinc-900/60 scrollbar-none">
          {INDIAN_ACTS.map((act) => {
            const isSelected = activeAct === act.id;
            return (
              <button
                key={act.id}
                onClick={() => handleActClick(act.id)}
                className={`px-2.5 py-0.5 text-[10px] font-medium rounded-full whitespace-nowrap transition-all duration-200 border ${
                  isSelected
                    ? "bg-emerald-950/80 text-emerald-400 border-emerald-800/60"
                    : "bg-zinc-900/40 text-zinc-400 border-zinc-800 hover:text-zinc-200"
                }`}
              >
                {act.id}
              </button>
            );
          })}
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col justify-center max-w-4xl w-full mx-auto px-4 py-8 sm:px-6 lg:px-8">
        
        <AnimatePresence mode="wait">
          
          {/* SCREEN 1: EMPTY / HOME STATE */}
          {screenState === "EMPTY" && (
            <motion.div
              key="empty-state"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.3 }}
              className="flex-1 flex flex-col justify-center items-center text-center space-y-8 my-auto"
            >
              {/* Central Glowing Shield / Scale Decor */}
              <div className="relative mb-2">
                <div className="absolute inset-0 bg-emerald-500/10 rounded-full blur-2xl w-24 h-24 mx-auto"></div>
                <div className="relative p-5 bg-gradient-to-b from-zinc-900 to-black rounded-2xl border border-zinc-800 shadow-xl shadow-black">
                  <Scale className="h-10 w-10 text-emerald-500 animate-pulse-slow" />
                </div>
              </div>

              {/* Title & Description */}
              <div className="space-y-3 max-w-xl">
                <h1 className="text-3xl sm:text-4xl md:text-5xl font-extrabold tracking-tight text-white leading-tight">
                  Ask any question across <br />
                  <span className="bg-gradient-to-r from-emerald-400 via-emerald-500 to-teal-400 bg-clip-text text-transparent">
                    7 major Indian Acts
                  </span>
                </h1>
                <p className="text-zinc-400 text-sm sm:text-base max-w-md mx-auto">
                  A high-precision intelligence terminal for CrPC, RTI, HMA, IT Act, CPA, ICA, and IPC.
                </p>
              </div>

              {/* Example Chips */}
              {/* <div className="w-full max-w-2xl space-y-3">
                <p className="text-[10px] font-mono tracking-widest text-zinc-500 uppercase">
                  Suggested inquiries
                </p>
                <div className="flex flex-col sm:flex-row flex-wrap gap-2 justify-center items-center">
                  {RECOMMENDED_CHIPS.map((chip, index) => (
                    <button
                      key={index}
                      onClick={() => handleChipClick(chip.text, chip.act)}
                      className="px-4 py-2.5 rounded-xl bg-[#0b0b0c] hover:bg-[#121214] border border-zinc-800 hover:border-zinc-700 text-xs text-zinc-300 hover:text-white transition-all text-left sm:text-center w-full sm:w-auto flex items-center space-x-2 group"
                    >
                      <Sparkles className="h-3 w-3 text-emerald-500/70 group-hover:text-emerald-400 transition-colors shrink-0" />
                      <span className="truncate">"{chip.text}"</span>
                    </button>
                  ))}
                </div>
              </div> */}
            </motion.div>
          )}

          {/* SCREEN 2: LOADING STATE */}
          {screenState === "LOADING" && (
            <motion.div
              key="loading-state"
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.98 }}
              transition={{ duration: 0.2 }}
              className="flex-1 flex flex-col justify-center items-center space-y-8 my-auto"
            >
              {/* Skeleton Answer Card */}
              <div className="w-full max-w-3xl bg-[#09090b] rounded-2xl border border-zinc-900 p-6 sm:p-8 space-y-6 shadow-2xl relative overflow-hidden">
                {/* Simulated Glow Shimmer */}
                <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-emerald-500 to-transparent animate-[shimmer_1.5s_infinite]"></div>
                
                <div className="flex items-center space-x-3">
                  {/* Avatar circle skeleton */}
                  <div className="h-6 w-6 rounded-full bg-zinc-800 animate-pulse"></div>
                  {/* Label title skeleton */}
                  <div className="h-4 w-28 rounded bg-zinc-800 animate-pulse"></div>
                </div>

                {/* Subheading skeleton line */}
                <div className="h-6 w-11/12 rounded bg-zinc-800/80 animate-pulse"></div>
                <div className="h-6 w-3/4 rounded bg-zinc-800/80 animate-pulse"></div>

                <hr className="border-zinc-900" />

                {/* Main answer text skeleton lines */}
                <div className="space-y-3.5 pt-2">
                  <div className="h-3 w-full rounded bg-zinc-900 animate-pulse"></div>
                  <div className="h-3 w-full rounded bg-zinc-900 animate-pulse"></div>
                  <div className="h-3 w-5/6 rounded bg-zinc-900 animate-pulse"></div>
                  <div className="h-3 w-full rounded bg-zinc-900 animate-pulse"></div>
                  <div className="h-3 w-4/6 rounded bg-zinc-900 animate-pulse"></div>
                </div>
              </div>

              {/* Loader Indicator Details */}
              <div className="flex flex-col items-center space-y-3">
                <span className="text-xs font-mono tracking-widest text-zinc-500 uppercase">
                  Searching across 7 acts...
                </span>
                <div className="flex space-x-1.5 items-center">
                  <div className="h-2 w-2 rounded-full bg-emerald-500 animate-[bounce_1.2s_infinite_100ms]"></div>
                  <div className="h-2 w-2 rounded-full bg-emerald-500 animate-[bounce_1.2s_infinite_200ms]"></div>
                  <div className="h-2 w-2 rounded-full bg-emerald-500 animate-[bounce_1.2s_infinite_300ms]"></div>
                </div>
              </div>
            </motion.div>
          )}

          {/* SCREEN 3: ANSWER STATE */}
          {screenState === "ANSWER" && currentResponse && (
            <motion.div
              key="answer-state"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.3 }}
              className="flex-1 flex flex-col justify-center space-y-6"
            >
              {/* Back / Clear controls above card */}
              <div className="flex items-center justify-between">
                <button
                  onClick={handleReset}
                  className="inline-flex items-center space-x-2 text-xs text-zinc-400 hover:text-white transition-colors"
                >
                  <ArrowLeft className="h-4 w-4" />
                  <span>Back to search</span>
                </button>

                <button
                  onClick={handleReset}
                  className="inline-flex items-center space-x-2 px-3 py-1.5 rounded-xl bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 hover:border-zinc-700 text-xs font-medium text-zinc-300 hover:text-white transition-all shadow-sm shadow-black"
                >
                  <RotateCcw className="h-3 w-3" />
                  <span>Ask another question</span>
                </button>
              </div>

              {/* Beautiful Answer Card Container */}
              <div className="w-full max-w-4xl bg-[#09090b] rounded-2xl border border-zinc-900 p-6 sm:p-8 space-y-6 shadow-2xl relative">
                
                {/* Section Indicator */}
                <div className="space-y-1">
                  <p className="text-[10px] font-mono tracking-widest text-zinc-500 uppercase">
                    Your Question
                  </p>
                  <h2 className="text-lg sm:text-xl font-bold tracking-tight text-white leading-relaxed">
                    {currentResponse.question}
                  </h2>
                </div>

                <hr className="border-zinc-900" />

                {/* Answer Content */}
                <div className="space-y-4 text-zinc-300 text-sm sm:text-base leading-relaxed font-sans">
                  {currentResponse.answer.split("\n\n").map((para, i) => (
                    <p key={i}>
                      {parseAnswerText(para)}
                    </p>
                  ))}
                </div>

                <hr className="border-zinc-900" />

                {/* Sources & Action bar footer */}
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 pt-2">
                  <div className="space-y-2">
                    <p className="text-[10px] font-mono tracking-widest text-zinc-500 uppercase">
                      Retrieved From
                    </p>
                    <div className="flex flex-wrap gap-1.5">
                      {currentResponse.retrievedFrom.map((source, idx) => (
                        <span
                          key={idx}
                          className="px-2.5 py-1 rounded bg-zinc-900 text-[11px] font-mono text-zinc-400 border border-zinc-800"
                        >
                          {source}
                        </span>
                      ))}
                    </div>
                  </div>

                  {/* <div className="self-end sm:self-center">
                    <button
                      onClick={handleReset}
                      className="inline-flex items-center space-x-2 px-4 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-xs font-semibold text-white transition-all shadow-lg shadow-emerald-950/20"
                    >
                      <Sparkles className="h-3 w-3" />
                      <span>Ask another question</span>
                    </button>
                  </div> */}
                </div>

              </div>
            </motion.div>
          )}

        </AnimatePresence>

        {/* Persistent Bottom Query Bar (always visible at bottom) */}
        <div className="mt-8">
          <form 
            onSubmit={onSubmit}
            className="relative flex items-center bg-[#09090b] rounded-2xl border border-zinc-800 focus-within:border-zinc-700 transition-all p-2 pr-3 shadow-xl"
          >
            {/* Left Decor Icon */}
            <div className="pl-3 pr-2 text-zinc-500">
              <Scale className="h-5 w-5 text-emerald-500/60" />
            </div>

            {/* Input Element */}
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Ask about Section 302, grounds for divorce, or contract breach..."
              disabled={screenState === "LOADING"}
              className="flex-1 bg-transparent text-sm text-zinc-100 placeholder-zinc-500 focus:outline-none py-3 px-1 disabled:opacity-50"
            />

            {/* Submit Action Controls */}
            <div className="flex items-center space-x-2 shrink-0">
              {/* Keyboard helper tag (only shown on md screens) */}
              <kbd className="hidden md:inline-flex items-center px-2 py-1 text-[10px] font-mono text-zinc-500 bg-zinc-900 border border-zinc-800 rounded">
                ⌘ K
              </kbd>

              {/* Dynamic Send Button */}
              <button
                type="submit"
                disabled={!searchQuery.trim() || screenState === "LOADING"}
                className={`p-2.5 rounded-xl transition-all flex items-center justify-center ${
                  !searchQuery.trim() || screenState === "LOADING"
                    ? "bg-zinc-900 text-zinc-600 cursor-not-allowed border border-zinc-800/40"
                    : "bg-white text-black hover:bg-zinc-200 active:scale-95 shadow-md"
                }`}
              >
                {screenState === "LOADING" ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <ArrowUp className="h-4 w-4 stroke-[3]" />
                )}
              </button>
            </div>
          </form>

          {/* Footer Branding info */}
          <div className="flex flex-col sm:flex-row items-center justify-between mt-4 px-2 text-[11px] text-zinc-500 gap-2">
            <p>LexAI Indian Legal Intelligence © 2024</p>
            <div className="flex items-center space-x-3 font-mono">
              <span>Press <kbd className="bg-zinc-900 px-1 border border-zinc-800 rounded">Enter ↵</kbd> to search</span>
              <span className="text-zinc-700">|</span>
              <a href="#" className="hover:text-zinc-300">Support</a>
              <span>·</span>
              <a href="#" className="hover:text-zinc-300">Privacy</a>
            </div>
          </div>
        </div>

      </main>
    </div>
  );
}
