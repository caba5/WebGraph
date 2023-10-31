use crate::{huffman_zuckerli::{HuffmanSymbolInfo, K_MAX_HUFFMAN_BITS, K_MAX_NUM_CONTEXTS, K_NUM_SYMBOLS}, bitstreams::BinaryReader, utils::encodings::{zuck_decode, K_ZUCK, I_ZUCK, J_ZUCK}};

use super::Huffman;

#[derive(Clone, Copy, Default, Debug)]
struct HuffmanDecoderInfo {
    nbits: u8,
    symbol: u8,
}

#[derive(Clone)]
pub struct HuffmanDecoder {
    info_: [[HuffmanDecoderInfo; 1 << K_MAX_HUFFMAN_BITS]; K_MAX_NUM_CONTEXTS],
}

impl Default for HuffmanDecoder {
    fn default() -> Self {
        Self { info_: [[HuffmanDecoderInfo::default(); 1 << K_MAX_HUFFMAN_BITS]; K_MAX_NUM_CONTEXTS] }
    }
}

impl Huffman for HuffmanDecoder {}

impl HuffmanDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    fn decode_symbol_n_bits(&self, info: &mut [HuffmanSymbolInfo], reader: &mut BinaryReader) {
        let ms = reader.read_int(8);
        for inf in info.iter_mut().take(ms as usize + 1) {
            inf.present = reader.read_int(1) as u8;
            if inf.present == 1 {
                inf.nbits = reader.read_int(3) as u8 + 1;
            }
        }
        for inf in info.iter_mut().skip(ms as usize + 1) {
            inf.present = 0;
        }
    }

    /// Computes the lookup table from bitstream bits to decoded symbol for the decoder.
    fn compute_decoder_table(&mut self, sym_info: &mut [HuffmanSymbolInfo], ctx: usize) {
        let mut cnt = 0;
        let mut s = 0;
        for (i, sym) in sym_info.iter().enumerate() {
            if sym.present == 0 {
                continue;
            }

            cnt += 1;
            s = i;
        }

        if cnt <= 1 {
            for dinfo in self.info_[ctx].iter_mut() {
                dinfo.nbits = sym_info[s].nbits;
                dinfo.symbol = s as u8;
            }
            return;
        }

        for (i, dinfo) in self.info_[ctx].iter_mut().enumerate() {
            let mut s = K_NUM_SYMBOLS;
            for (j, sym) in sym_info.iter_mut().enumerate() {
                if sym.present == 0 {
                    continue;
                }
                if (i as u16 & ((1 << sym.nbits) - 1)) == sym.bits {
                    sym.present = 0;
                    s = j;
                    break;
                }
            }
            // assert_ne!(s, K_NUM_SYMBOLS, "Invalid table");
            if s != K_NUM_SYMBOLS {
                dinfo.nbits = sym_info[s].nbits;
                dinfo.symbol = s as u8;
            }
        }
    }

    #[inline(always)]
    pub fn decode_headers(&mut self, bin_reader: &mut BinaryReader, num_contexts: usize) {
        assert!(num_contexts < K_MAX_NUM_CONTEXTS, "The number of contexts has to be smaller than {K_MAX_NUM_CONTEXTS}");
        for ctx in 0..num_contexts {
            let mut symbol_info = [HuffmanSymbolInfo::default(); K_NUM_SYMBOLS];
            self.decode_symbol_n_bits(&mut symbol_info, bin_reader);
            HuffmanDecoder::compute_symbol_bits(&mut symbol_info);
            self.compute_decoder_table(&mut symbol_info, ctx);
        }
    }

    #[inline(always)]
    pub fn read_next(&mut self, bin_reader: &mut BinaryReader, ctx: usize) -> usize {
        let mut curr_len = 0;
        let mut curr_code = 0;

        for _ in 0..K_MAX_HUFFMAN_BITS {
            debug_assert!(bin_reader.get_position() < bin_reader.is.len() * 8);
            curr_len += 1;
            curr_code = (curr_code << 1) | bin_reader.read_int(1);
            if self.info_[ctx][curr_code as usize].nbits == curr_len as u8 {
                return zuck_decode(self.info_[ctx][curr_code as usize].symbol as usize, bin_reader, K_ZUCK, I_ZUCK, J_ZUCK);
            }
        }
        
        panic!("Malformed code")
    }
}