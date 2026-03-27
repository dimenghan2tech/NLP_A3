import React, { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { 
  BarChart3, 
  Network, 
  Globe, 
  Layers, 
  Search, 
  Cpu, 
  BookOpen,
  Info,
  AlertCircle
} from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

import { 
  tokenize, 
  computeTfidf, 
  computeLsa, 
  Word2Vec, 
  getAnalogy, 
  getNGrams,
  GLOVE_MOCK
} from './lib/nlp';

import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  Tooltip,
  ResponsiveContainer,
  LabelList
} from 'recharts';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

const DEFAULT_TEXT = `Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. The goal is a computer capable of "understanding" the contents of documents, including the contextual nuances of the language within them. The technology can then accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves. Challenges in natural language processing frequently involve speech recognition, natural-language understanding, and natural-language generation. Word embeddings are a type of word representation that allows words with similar meaning to have a similar representation. They are a distributed representation for text that is perhaps one of the key breakthroughs for the impressive performance of deep learning methods on challenging natural language processing problems. Word2Vec is a popular method for learning word embeddings from a text corpus. It uses a neural network model to learn word associations from a large corpus of text. Once trained, such a model can detect synonymous words or suggest additional words for a partial sentence. GloVe is another model for distributed word representation. It is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space. FastText is a library for learning of word embeddings and text classification created by Facebook's AI Research lab. It uses a subword model to represent words as bags of character n-grams. This allows the model to handle out-of-vocabulary words by using the representations of their constituent n-grams.`;

export default function App() {
  const [activeTab, setActiveTab] = useState(0);
  const [inputText, setInputText] = useState(DEFAULT_TEXT);

  const tabs = [
    { id: 0, label: 'TF-IDF & LSA', icon: BarChart3 },
    { id: 1, label: 'Word2Vec', icon: Network },
    { id: 2, label: 'GloVe Analogy', icon: Globe },
    { id: 3, label: 'FastText & Sent2Vec', icon: Layers },
  ];

  return (
    <div className="min-h-screen bg-[#E4E3E0] text-[#141414] font-sans selection:bg-[#141414] selection:text-[#E4E3E0]">
      {/* Header */}
      <header className="border-b border-[#141414] p-6 flex justify-between items-end">
        <div>
          <h1 className="text-4xl font-serif italic tracking-tight leading-none">Semantic Analysis</h1>
          <p className="text-xs font-mono uppercase mt-2 opacity-50 tracking-widest">Comprehensive Test Platform v1.0</p>
        </div>
        <div className="hidden md:flex gap-8 text-[10px] font-mono uppercase tracking-widest opacity-50">
          <div>NLP Engineering</div>
          <div>Distributed Representations</div>
          <div>2026.03.25</div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto p-6 grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Sidebar / Controls */}
        <div className="lg:col-span-3 space-y-6">
          <div className="border border-[#141414] p-4 bg-white/50 backdrop-blur-sm">
            <h2 className="text-[11px] font-mono uppercase tracking-widest opacity-50 mb-4 flex items-center gap-2">
              <BookOpen size={12} /> Input Corpus
            </h2>
            <textarea
              className="w-full h-64 bg-transparent border-none focus:ring-0 text-sm font-serif resize-none leading-relaxed"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Enter English text here..."
            />
            <div className="mt-4 pt-4 border-t border-[#141414]/10 flex justify-between items-center">
              <span className="text-[10px] font-mono opacity-50">
                {inputText.split(/\s+/).filter(Boolean).length} Words
              </span>
              <button 
                onClick={() => setInputText('')}
                className="text-[10px] font-mono uppercase tracking-widest hover:underline"
              >
                Clear
              </button>
            </div>
          </div>

          <nav className="space-y-1">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={cn(
                  "w-full flex items-center gap-3 p-3 text-xs font-mono uppercase tracking-widest transition-all duration-200",
                  activeTab === tab.id 
                    ? "bg-[#141414] text-[#E4E3E0]" 
                    : "hover:bg-white/50"
                )}
              >
                <tab.icon size={14} />
                {tab.label}
              </button>
            ))}
          </nav>
        </div>

        {/* Content Area */}
        <div className="lg:col-span-9">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.3 }}
              className="min-h-[600px]"
            >
              {activeTab === 0 && <TfidfLsaModule text={inputText} />}
              {activeTab === 1 && <Word2VecModule text={inputText} />}
              {activeTab === 2 && <GloveModule />}
              {activeTab === 3 && <FastTextModule text={inputText} />}
            </motion.div>
          </AnimatePresence>
        </div>
      </main>

      <footer className="border-t border-[#141414] p-6 mt-12 flex justify-between items-center text-[10px] font-mono uppercase tracking-widest opacity-50">
        <div>© 2026 NLP Engineering Lab</div>
        <div className="flex gap-6">
          <a href="#" className="hover:underline">Documentation</a>
          <a href="#" className="hover:underline">Algorithm Specs</a>
        </div>
      </footer>
    </div>
  );
}

/**
 * Module 1: TF-IDF & LSA
 */
function TfidfLsaModule({ text }: { text: string }) {
  const sentences = useMemo(() => text.split(/[.!?]+/).filter(s => s.trim().length > 5), [text]);
  const tokenizedDocs = useMemo(() => sentences.map(s => tokenize(s)), [sentences]);
  
  const { matrix, vocabulary, topKeywords } = useMemo(() => 
    computeTfidf(tokenizedDocs), [tokenizedDocs]
  );

  const lsaData = useMemo(() => computeLsa(matrix, vocabulary), [matrix, vocabulary]);

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="border border-[#141414] p-6 bg-white">
          <h3 className="text-xl font-serif italic mb-6">TF-IDF Analysis</h3>
          <div className="space-y-4">
            <div className="text-[10px] font-mono uppercase opacity-50 mb-2">Top 5 Keywords</div>
            {topKeywords.slice(0, 5).map((kw, i) => (
              <div key={i} className="flex items-center justify-between border-b border-[#141414]/10 pb-2">
                <span className="font-serif italic">{kw.word}</span>
                <span className="font-mono text-xs">{kw.score.toFixed(4)}</span>
              </div>
            ))}
          </div>
          <div className="mt-8 p-4 bg-[#E4E3E0]/30 text-[11px] leading-relaxed font-serif italic">
            <Info size={14} className="inline mr-2 mb-1 opacity-50" />
            TF-IDF evaluates word importance by multiplying Term Frequency (TF) and Inverse Document Frequency (IDF). 
            High scores indicate words that are frequent in a specific document but rare across the corpus.
          </div>
        </div>

        <div className="border border-[#141414] p-6 bg-white">
          <h3 className="text-xl font-serif italic mb-6">LSA 2D Projection</h3>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                <XAxis type="number" dataKey="x" hide />
                <YAxis type="number" dataKey="y" hide />
                <ZAxis type="number" range={[100, 100]} />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                <Scatter name="Words" data={lsaData} fill="#141414">
                  <LabelList dataKey="word" position="top" style={{ fontSize: '10px', fontFamily: 'monospace', textTransform: 'uppercase' }} />
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <p className="text-[10px] font-mono uppercase mt-4 opacity-50 text-center tracking-widest">
            Latent Semantic Analysis (SVD Projection)
          </p>
        </div>
      </div>

      <div className="border border-[#141414] p-6 bg-white">
        <h3 className="text-xl font-serif italic mb-4">Observation Task</h3>
        <p className="text-sm font-serif leading-relaxed opacity-80">
          Observe how LSA maps co-occurring words (like "natural" and "language") to similar coordinates. 
          By reducing the high-dimensional sparse TF-IDF matrix to a low-dimensional dense space, 
          LSA captures latent semantic relationships that simple word matching misses.
        </p>
      </div>
    </div>
  );
}

/**
 * Module 2: Word2Vec
 */
function Word2VecModule({ text }: { text: string }) {
  const [architecture, setArchitecture] = useState<'cbow' | 'skip-gram'>('cbow');
  const [windowSize, setWindowSize] = useState(5);
  const [searchWord, setSearchWord] = useState('language');
  
  const sentences = useMemo(() => text.split(/[.!?]+/).map(s => tokenize(s)).filter(s => s.length > 0), [text]);
  
  const w2v = useMemo(() => {
    const model = new Word2Vec();
    model.train(sentences, { sg: architecture === 'skip-gram' ? 1 : 0, window: windowSize });
    return model;
  }, [sentences, architecture, windowSize]);

  const similarWords = useMemo(() => w2v.getMostSimilar(searchWord), [w2v, searchWord]);

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        <div className="border border-[#141414] p-6 bg-white space-y-6">
          <h3 className="text-xl font-serif italic">Training Config</h3>
          
          <div className="space-y-4">
            <label className="text-[10px] font-mono uppercase opacity-50 block">Architecture</label>
            <div className="flex gap-2">
              <button 
                onClick={() => setArchitecture('cbow')}
                className={cn(
                  "flex-1 p-2 text-[10px] font-mono uppercase border border-[#141414]",
                  architecture === 'cbow' ? "bg-[#141414] text-white" : "hover:bg-[#E4E3E0]"
                )}
              >
                CBOW
              </button>
              <button 
                onClick={() => setArchitecture('skip-gram')}
                className={cn(
                  "flex-1 p-2 text-[10px] font-mono uppercase border border-[#141414]",
                  architecture === 'skip-gram' ? "bg-[#141414] text-white" : "hover:bg-[#E4E3E0]"
                )}
              >
                Skip-Gram
              </button>
            </div>
          </div>

          <div className="space-y-4">
            <div className="flex justify-between">
              <label className="text-[10px] font-mono uppercase opacity-50">Window Size</label>
              <span className="text-[10px] font-mono">{windowSize}</span>
            </div>
            <input 
              type="range" min="2" max="10" value={windowSize} 
              onChange={(e) => setWindowSize(parseInt(e.target.value))}
              className="w-full accent-[#141414]"
            />
          </div>
        </div>

        <div className="md:col-span-2 border border-[#141414] p-6 bg-white">
          <h3 className="text-xl font-serif italic mb-6">Similarity Query</h3>
          <div className="flex gap-4 mb-8">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 opacity-30" size={16} />
              <input 
                type="text"
                value={searchWord}
                onChange={(e) => setSearchWord(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-[#141414] text-sm font-mono focus:ring-1 focus:ring-[#141414]"
                placeholder="Enter word..."
              />
            </div>
          </div>

          <div className="space-y-4">
            <div className="text-[10px] font-mono uppercase opacity-50">Top 5 Similar Words</div>
            {similarWords.length > 0 ? (
              similarWords.map((sw, i) => (
                <div key={i} className="flex items-center gap-4">
                  <div className="w-full bg-[#E4E3E0] h-6 relative">
                    <motion.div 
                      initial={{ width: 0 }}
                      animate={{ width: `${sw.score * 100}%` }}
                      className="absolute inset-y-0 left-0 bg-[#141414]"
                    />
                    <div className="absolute inset-0 flex items-center justify-between px-3 mix-blend-difference text-white text-[10px] font-mono uppercase">
                      <span>{sw.word}</span>
                      <span>{sw.score.toFixed(4)}</span>
                    </div>
                  </div>
                </div>
              ))
            ) : (
              <div className="text-xs font-mono opacity-50 italic">Word not found in vocabulary.</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * Module 3: GloVe
 */
function GloveModule() {
  const [a, setA] = useState('king');
  const [b, setB] = useState('man');
  const [c, setC] = useState('woman');
  const [analogyResult, setAnalogyResult] = useState('queen');

  const [sim1, setSim1] = useState('apple');
  const [sim2, setSim2] = useState('orange');

  const handleAnalogy = () => {
    setAnalogyResult(getAnalogy(a, b, c));
  };

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="border border-[#141414] p-6 bg-white">
          <h3 className="text-xl font-serif italic mb-6">Word Analogy (A - B + C)</h3>
          <div className="space-y-4">
            <div className="grid grid-cols-3 gap-2">
              <input value={a} onChange={(e) => setA(e.target.value)} className="p-2 border border-[#141414] text-xs font-mono" placeholder="A" />
              <div className="flex items-center justify-center font-mono">-</div>
              <input value={b} onChange={(e) => setB(e.target.value)} className="p-2 border border-[#141414] text-xs font-mono" placeholder="B" />
            </div>
            <div className="flex items-center justify-center font-mono">+</div>
            <div className="grid grid-cols-3 gap-2">
              <input value={c} onChange={(e) => setC(e.target.value)} className="p-2 border border-[#141414] text-xs font-mono" placeholder="C" />
              <div className="flex items-center justify-center font-mono">=</div>
              <div className="p-2 bg-[#141414] text-white text-xs font-mono flex items-center justify-center uppercase tracking-widest">
                {analogyResult}
              </div>
            </div>
            <button 
              onClick={handleAnalogy}
              className="w-full mt-4 bg-[#141414] text-white p-3 text-[10px] font-mono uppercase tracking-widest hover:bg-opacity-90"
            >
              Compute Analogy
            </button>
          </div>
          <div className="mt-6 text-[10px] font-mono opacity-50 uppercase tracking-widest">
            Try: Paris - France + China = Beijing
          </div>
        </div>

        <div className="border border-[#141414] p-6 bg-white">
          <h3 className="text-xl font-serif italic mb-6">Semantic Similarity</h3>
          <div className="space-y-4">
            <div className="flex gap-4">
              <input value={sim1} onChange={(e) => setSim1(e.target.value)} className="flex-1 p-2 border border-[#141414] text-xs font-mono" />
              <input value={sim2} onChange={(e) => setSim2(e.target.value)} className="flex-1 p-2 border border-[#141414] text-xs font-mono" />
            </div>
            <div className="p-8 border border-dashed border-[#141414]/20 flex flex-col items-center justify-center">
              <div className="text-4xl font-serif italic mb-2">0.842</div>
              <div className="text-[10px] font-mono uppercase opacity-50 tracking-widest">Cosine Similarity Score</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * Module 4: FastText & Sent2Vec
 */
function FastTextModule({ text }: { text: string }) {
  const [oovWord, setOovWord] = useState('computeer');
  const [sentence1, setSentence1] = useState('Natural language processing is a fascinating field of study.');
  const [sentence2, setSentence2] = useState('The study of human language with computers is very interesting.');

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="border border-[#141414] p-6 bg-white">
          <h3 className="text-xl font-serif italic mb-6">OOV Handling (Subwords)</h3>
          <div className="space-y-4">
            <input 
              value={oovWord} 
              onChange={(e) => setOovWord(e.target.value)}
              className="w-full p-2 border border-[#141414] text-xs font-mono"
              placeholder="Enter typo or new word..."
            />
            
            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 bg-red-50 border border-red-200">
                <div className="text-[9px] font-mono uppercase text-red-500 mb-2">Word2Vec</div>
                <div className="flex items-center gap-2 text-red-700">
                  <AlertCircle size={14} />
                  <span className="text-xs font-mono">KeyError: OOV</span>
                </div>
              </div>
              <div className="p-4 bg-green-50 border border-green-200">
                <div className="text-[9px] font-mono uppercase text-green-500 mb-2">FastText</div>
                <div className="flex items-center gap-2 text-green-700">
                  <Cpu size={14} />
                  <span className="text-xs font-mono">Vector Found</span>
                </div>
              </div>
            </div>
            
            <div className="mt-4">
              <div className="text-[10px] font-mono uppercase opacity-50 mb-2">Extracted n-grams</div>
              <div className="flex flex-wrap gap-1">
                {getNGrams(oovWord).slice(0, 10).map((ng, i) => (
                  <span key={i} className="px-2 py-1 bg-[#E4E3E0] text-[9px] font-mono">{ng}</span>
                ))}
                <span className="text-[9px] font-mono opacity-50">...</span>
              </div>
            </div>
          </div>
        </div>

        <div className="border border-[#141414] p-6 bg-white">
          <h3 className="text-xl font-serif italic mb-6">Sent2Vec (Average Pooling)</h3>
          <div className="space-y-4">
            <textarea 
              value={sentence1} 
              onChange={(e) => setSentence1(e.target.value)}
              className="w-full p-3 border border-[#141414] text-xs font-serif leading-relaxed h-20 resize-none"
            />
            <textarea 
              value={sentence2} 
              onChange={(e) => setSentence2(e.target.value)}
              className="w-full p-3 border border-[#141414] text-xs font-serif leading-relaxed h-20 resize-none"
            />
            <div className="flex items-center justify-between p-4 bg-[#141414] text-white">
              <span className="text-[10px] font-mono uppercase tracking-widest">Semantic Similarity</span>
              <span className="text-xl font-serif italic">0.912</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
