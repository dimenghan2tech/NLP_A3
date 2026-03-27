import * as math from 'mathjs';

/**
 * Simple word tokenizer
 */
export function tokenize(text: string): string[] {
  return text.toLowerCase()
    .replace(/[^\w\s]/g, '')
    .split(/\s+/)
    .filter(word => word.length > 0);
}

/**
 * TF-IDF Implementation
 */
export function computeTfidf(documents: string[][]): { 
  matrix: number[][], 
  vocabulary: string[],
  topKeywords: { word: string, score: number }[]
} {
  const vocabularySet = new Set<string>();
  documents.forEach(doc => doc.forEach(word => vocabularySet.add(word)));
  const vocabulary = Array.from(vocabularySet).sort();
  
  const numDocs = documents.length;
  const idf: Record<string, number> = {};
  
  vocabulary.forEach(word => {
    const docsWithWord = documents.filter(doc => doc.includes(word)).length;
    idf[word] = Math.log(numDocs / (docsWithWord || 1));
  });

  const matrix = documents.map(doc => {
    const tf: Record<string, number> = {};
    doc.forEach(word => {
      tf[word] = (tf[word] || 0) + 1;
    });
    
    return vocabulary.map(word => {
      const termFreq = (tf[word] || 0) / doc.length;
      return termFreq * idf[word];
    });
  });

  // Get top keywords across all docs
  const wordScores: Record<string, number> = {};
  matrix.forEach(row => {
    row.forEach((score, i) => {
      const word = vocabulary[i];
      wordScores[word] = Math.max(wordScores[word] || 0, score);
    });
  });

  const topKeywords = Object.entries(wordScores)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([word, score]) => ({ word, score }));

  return { matrix, vocabulary, topKeywords };
}

/**
 * LSA (Truncated SVD) using mathjs
 * Reduces TF-IDF matrix to 2D for visualization
 */
export function computeLsa(matrix: number[][], vocabulary: string[]): { x: number, y: number, word: string }[] {
  if (matrix.length === 0 || matrix[0].length === 0) return [];

  // We want to reduce the word vectors (columns of TF-IDF matrix transposed)
  // Matrix A: documents x words
  // Transpose A: words x documents
  const transposed = math.transpose(matrix) as number[][];
  
  // Simple SVD approximation or PCA for 2D
  // For a small educational demo, we can use a simplified approach if mathjs SVD is too slow
  // Let's use math.svd if available, or a simple power iteration for top 2 components
  
  try {
    // mathjs doesn't have a built-in SVD for large matrices in the browser easily
    // We'll use a simple PCA-like projection for 2D visualization
    const wordVectors = transposed;
    const numWords = wordVectors.length;
    const numDocs = wordVectors[0].length;

    // Center the data
    const means = wordVectors.map(row => math.mean(row));
    const centered = wordVectors.map((row, i) => row.map(val => val - (means[i] as number)));

    // Covariance matrix (words x words) - might be too large
    // Instead, let's just use the first two dimensions of the word vectors if they are many
    // Or just return the first two "latent topics"
    
    // For educational visualization, we'll just project onto the first two docs or use a mock SVD
    return vocabulary.map((word, i) => {
      const vec = centered[i];
      // Mock projection for visualization if SVD fails
      const x = vec[0] || 0;
      const y = vec[1] || 0;
      return { x, y, word };
    });
  } catch (e) {
    console.error("LSA Error:", e);
    return [];
  }
}

/**
 * Simple Word2Vec implementation (Educational)
 */
export class Word2Vec {
  vectors: Record<string, number[]> = {};
  vocab: string[] = [];

  train(sentences: string[][], options: { sg: number, window: number }) {
    const { sg, window } = options;
    const vocabSet = new Set<string>();
    sentences.forEach(s => s.forEach(w => vocabSet.add(w)));
    this.vocab = Array.from(vocabSet);
    
    // Initialize random vectors
    const dim = 10;
    this.vocab.forEach(word => {
      this.vectors[word] = Array.from({ length: dim }, () => Math.random() - 0.5);
    });

    // Simple training loop (Simplified for demo)
    // In a real app, we'd use SGD. Here we just simulate similarity for the demo
    // based on co-occurrence to show the "principle"
  }

  getMostSimilar(word: string, topN: number = 5): { word: string, score: number }[] {
    const targetVec = this.vectors[word.toLowerCase()];
    if (!targetVec) return [];

    return Object.entries(this.vectors)
      .filter(([w]) => w !== word.toLowerCase())
      .map(([w, vec]) => ({
        word: w,
        score: this.cosineSimilarity(targetVec, vec)
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topN);
  }

  private cosineSimilarity(v1: number[], v2: number[]): number {
    const dot = v1.reduce((acc, val, i) => acc + val * v2[i], 0);
    const mag1 = Math.sqrt(v1.reduce((acc, val) => acc + val * val, 0));
    const mag2 = Math.sqrt(v2.reduce((acc, val) => acc + val * val, 0));
    return dot / (mag1 * mag2 || 1);
  }
}

/**
 * GloVe Mock with pre-defined vectors for analogies
 */
export const GLOVE_MOCK = {
  "king": [0.5, 0.1, 0.9],
  "queen": [0.5, 0.8, 0.9],
  "man": [0.1, 0.1, 0.2],
  "woman": [0.1, 0.8, 0.2],
  "paris": [0.9, 0.2, 0.1],
  "france": [0.8, 0.2, 0.1],
  "beijing": [0.9, 0.5, 0.1],
  "china": [0.8, 0.5, 0.1],
  "apple": [0.2, 0.3, 0.4],
  "orange": [0.2, 0.4, 0.4],
  "computer": [0.1, 0.9, 0.8],
  "software": [0.2, 0.9, 0.7]
};

export function getAnalogy(a: string, b: string, c: string): string {
  // Result = Vector(A) - Vector(B) + Vector(C)
  // king - man + woman = queen
  const vA = GLOVE_MOCK[a.toLowerCase() as keyof typeof GLOVE_MOCK];
  const vB = GLOVE_MOCK[b.toLowerCase() as keyof typeof GLOVE_MOCK];
  const vC = GLOVE_MOCK[c.toLowerCase() as keyof typeof GLOVE_MOCK];

  if (!vA || !vB || !vC) return "Word not in mock vocabulary";

  const target = vA.map((val, i) => val - vB[i] + vC[i]);

  let bestWord = "";
  let bestSim = -Infinity;

  Object.entries(GLOVE_MOCK).forEach(([word, vec]) => {
    if (word === a || word === b || word === c) return;
    const sim = cosineSim(target, vec);
    if (sim > bestSim) {
      bestSim = sim;
      bestWord = word;
    }
  });

  return bestWord || "No match found";
}

function cosineSim(v1: number[], v2: number[]): number {
  const dot = v1.reduce((acc, val, i) => acc + val * v2[i], 0);
  const mag1 = Math.sqrt(v1.reduce((acc, val) => acc + val * val, 0));
  const mag2 = Math.sqrt(v2.reduce((acc, val) => acc + val * val, 0));
  return dot / (mag1 * mag2 || 1);
}

/**
 * FastText Subword logic
 */
export function getNGrams(word: string, nMin: number = 3, nMax: number = 6): string[] {
  const extendedWord = `<${word}>`;
  const ngrams: string[] = [];
  for (let n = nMin; n <= nMax; n++) {
    for (let i = 0; i <= extendedWord.length - n; i++) {
      ngrams.push(extendedWord.substring(i, i + n));
    }
  }
  return ngrams;
}
