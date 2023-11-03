use std::mem::take;

use crate::{huffman_zuckerli::{HuffmanSymbolInfo, K_MAX_HUFFMAN_BITS, K_MAX_NUM_CONTEXTS, K_NUM_SYMBOLS}, bitstreams::BinaryWriterBuilder, utils::encodings::{zuck_encode, K_ZUCK, I_ZUCK, J_ZUCK}};

use super::Huffman;

pub struct HuffmanEncoder {
    info_: [[HuffmanSymbolInfo; 1 << K_MAX_HUFFMAN_BITS]; K_MAX_NUM_CONTEXTS],
}

impl Default for HuffmanEncoder {
    fn default() -> Self {
        Self { info_: [[HuffmanSymbolInfo::default(); 1 << K_MAX_HUFFMAN_BITS]; K_MAX_NUM_CONTEXTS] }
    }
}

impl Huffman for HuffmanEncoder {}


impl HuffmanEncoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Computes the optimal number of bits for each symbol given the input
    /// distribution. Uses a (quadratic version) of the package-merge/coin-collector
    /// algorithm. 
    fn compute_symbol_num_bits(&mut self, histo: &[u64], ctx: usize) {
        // Mark the present/missing symbols.
        let mut nzsym = 0;

        for (i, &value) in histo.iter().enumerate() {
            if value == 0 {
                continue;
            }
            self.info_[ctx][i].present = 1;
            nzsym += 1;
        }

        if nzsym <= 1 {
            for x in self.info_[ctx].iter_mut() {
                if x.present == 1 {
                    x.nbits = 1;
                }
            }

            return;
        }

        // Create a list of symbols for any given cost.
        let mut bags = vec![Vec::<(u64, Vec<u8>)>::default(); K_MAX_HUFFMAN_BITS];
        for bag in bags.iter_mut() {
            for (s, inf) in self.info_[ctx].iter_mut().enumerate() {
                if inf.present == 0 {
                    continue;
                }

                let sym = vec![s as u8; 1];
                bag.push((histo[s], sym));
            }
        }

        // Pair up symbols (or groups of symbols) of a given bit-length to create
        // symbols of the following bit-length, creating pairs by merging (groups of)
        // symbols consecutively in increasing order of cost.
        for i in 0..(K_MAX_HUFFMAN_BITS - 1) {
            bags[i].sort();

            let mut j = 0;
            while j + 1 < bags[i].len() {
                let nf = bags[i][j].0 + bags[i][j + 1].0;
                
                let mut nsym = take(&mut bags[i][j].1);
                // swap(&mut nsym, &mut bags[i][j].1);

                for &x in bags[i][j + 1].1.iter() {
                    nsym.push(x);
                }

                bags[i + 1].push((nf, nsym));

                j += 2;
            }
        }
        bags[K_MAX_HUFFMAN_BITS - 1].sort();

        // In the groups of symbols for the highest bit length we need to select the
        // last 2*num_symbols-2 groups, and assign to each symbol one bit of cost for
        // each of its occurrences in these groups.
        for i in 0..(2 * nzsym - 2) {
            let b = &bags[K_MAX_HUFFMAN_BITS - 1][i];
            for &x in b.1.iter() {
                self.info_[ctx][x as usize].nbits += 1;
            }
        }

        // In a properly-constructed set of lengths for a set of symbols, the sum
        // across the symbols of 2^-sym_length equals 1.
        let mut cost_check = 0;
        for inf in self.info_[ctx].iter() {
            if inf.present == 0 {
                continue;
            }
            
            cost_check += 1 << (K_MAX_HUFFMAN_BITS - inf.nbits as usize);
        }
        
        assert!(cost_check == 1 << K_MAX_HUFFMAN_BITS);
    }

    /// Very simple encoding: number of symbols (8 bits) followed by, for each
    /// symbol, 1 bit for presence/absence, and 3 bits for symbol length if present
    fn encode_symbol_n_bits(&mut self, writer: &mut BinaryWriterBuilder, ctx: usize) {
        let mut ms = 0;
        for (i, inf) in self.info_[ctx].iter().enumerate() {
            if inf.present == 1 {
                ms = i;
            }
        }

        writer.push_bits(ms as u64, 8);
        
        for inf in self.info_[ctx].iter().take(ms + 1) {
            if inf.present == 1 {
                writer.push_bits(1, 1);
                writer.push_bits(inf.nbits as u64 - 1, 3);
            } else {
                writer.push_bits(0, 1);
            }
        }
    }

    pub fn init(&mut self, integers: &[Vec<usize>], bin_writer: &mut BinaryWriterBuilder) {
        let num_contexts = integers.len();
        assert!(num_contexts < K_MAX_NUM_CONTEXTS, "The number of contexts has to be smaller than {K_MAX_NUM_CONTEXTS}");

        let mut histograms = vec![[0; K_NUM_SYMBOLS]; num_contexts];

        // Compute the frequencies
        for (ctx, ints) in integers.iter().enumerate() {
            for int in ints {
                let (token, _, _) = zuck_encode(*int, K_ZUCK, I_ZUCK, J_ZUCK);
                histograms[ctx][token] += 1;
            }
        }

        // Compute and encode symbol length and bits for each distinct symbol
        for (ctx, hist) in histograms.iter().enumerate() {
            self.compute_symbol_num_bits(hist, ctx);
            HuffmanEncoder::compute_symbol_bits(&mut self.info_[ctx]);
            self.encode_symbol_n_bits(bin_writer, ctx);
        }

        // Encode the actual data
        // for (ctx, ints) in integers.iter().enumerate() {
        //     for int in ints {
        //         let (token, nbits, bits) = zuck_encode(*int, K_ZUCK, I_ZUCK, J_ZUCK);
        //         bin_writer.push_bits(info[ctx][token].bits as u64, info[ctx][token].nbits as u64);
        //         bin_writer.push_bits(bits as u64, nbits as u64);
        //     }
        // }
    }

    #[inline(always)]
    pub fn write_next(&self, value: usize, bin_writer: &mut BinaryWriterBuilder, ctx: usize) -> usize {
        let (token, nbits, bits) = zuck_encode(value, K_ZUCK, I_ZUCK, J_ZUCK);
        assert!(self.info_[ctx][token].present == 1, "Unknown value {value}");
        bin_writer.push_bits(self.info_[ctx][token].bits as u64, self.info_[ctx][token].nbits as u64);
        bin_writer.push_bits(bits as u64, nbits as u64);
        self.info_[ctx][token].nbits as usize
    }
}