pub mod huffman_encoder;
pub mod huffman_decoder;

pub const K_MAX_HUFFMAN_BITS: usize = 8;
pub const K_NUM_SYMBOLS: usize = 256;
pub const K_MAX_NUM_CONTEXTS: usize = 256;

#[derive(Clone, Copy, Default, Debug)]
pub struct HuffmanSymbolInfo {
    present: u8,
    nbits: u8,
    bits: u16,
}

pub trait Huffman {
    /// For a given array of HuffmanSymbolInfo, where only the `present` and `nbits`
    /// fields are set, fill up the `bits` field by building a Canonical Huffman code.
    fn compute_symbol_bits(info: &mut [HuffmanSymbolInfo]) {
        let mut syms = [(u8::default(), u8::default()); K_NUM_SYMBOLS];
        let mut present_symbols = 0;
        
        for (i, inf) in info.iter().enumerate() {
            if inf.present == 0 {
                continue;
            }

            syms[present_symbols] = (inf.nbits, i as u8);
            present_symbols += 1;
        }

        syms[0..present_symbols].sort();

        let mut x = 0;

        for (s, sym) in syms.into_iter().take(present_symbols).enumerate() {
            info[sym.1 as usize].bits = x;
            x += 1;
            if s + 1 != present_symbols {
                x <<= syms[s + 1].0 - sym.0;
            }
        }
    }
}

#[cfg(test)]
mod tests;