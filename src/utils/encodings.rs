use crate::bitstreams::{BinaryReader, BinaryWriter, tables::{GAMMAS, ZETAS_3}};
use crate::utils::EncodingType;

pub trait UniversalCode {
    fn read_next(reader: &mut BinaryReader, zk: Option<u64>) -> u64;
    fn write_next(writer: &mut BinaryWriter, x: u64, zk: Option<u64>) -> u64;
    fn to_encoding_type() -> EncodingType;
}

pub struct UnaryCode;

impl UniversalCode for UnaryCode {
    #[inline(always)]
    fn read_next(reader: &mut BinaryReader, _zk: Option<u64>) -> u64 {
        assert!(reader.fill < 64);

        if reader.fill < 16 {
            reader.refill();
        }

        let mut x = u32::leading_zeros((reader.current as u32) << (32 - reader.fill));
        if x < reader.fill as u32{
            reader.read_bits += x as usize + 1;
            reader.fill -= x as usize + 1;
            return x as u64;
        }

        x = reader.fill as u32;
        let mut read = reader.read();

        if read.is_ok() {
            reader.current = read.unwrap();
            while reader.current == 0 && read.is_ok() {
                x += 8;
                read = reader.read();
                if let Ok(r) = read {
                    reader.current = r;
                }
            }
        }

        reader.fill = (63 - u64::leading_zeros(reader.current)) as usize;
        x += 7 - reader.fill as u32;
        reader.read_bits += x as usize + 1;
        x as u64
    }

    #[inline(always)]
    fn write_next(writer: &mut BinaryWriter, x: u64, _zk: Option<u64>) -> u64 {
        if x < writer.free as u64 {
            return writer.write_in_current(1, x + 1);
        }

        let shift = writer.free;
        let x = x - shift as u64;

        writer.written_bits += shift;
        writer.write(writer.current);
        writer.free = 8;
        writer.current = 0;
        
        let mut i = x >> 3;

        writer.written_bits += x as usize & 0x7FFFFFF8;
        
        while i != 0 {
            writer.write(0);
            i -= 1;
        }

        writer.write_in_current(1, (x & 7) + 1);

        x + shift as u64 + 1
    }

    #[inline(always)]
    fn to_encoding_type() -> EncodingType {
        EncodingType::UNARY
    }
}

pub struct GammaCode;

impl UniversalCode for GammaCode {
    #[inline(always)]
    fn read_next(reader: &mut BinaryReader, _zk: Option<u64>) -> u64 {
        if reader.fill >= 16 || reader.refill() >= 16 {
            let precomp = GAMMAS[reader.current as usize >> (reader.fill - 16) & 0xFFFF];
    
            if precomp.1 != 0 {
                reader.read_bits += precomp.1 as usize;
                reader.fill -= precomp.1 as usize;

                return precomp.0 as u64;
            }
        }

        let msb = UnaryCode::read_next(reader, None);
        ((1 << msb) | reader.read_int(msb)) - 1
    }

    #[inline(always)]
    fn write_next(writer: &mut BinaryWriter, x: u64, _zk: Option<u64>) -> u64 {
        assert!(x < u64::MAX);
        // if x < MAX_PRECOMPUTED TODO

        let x = x + 1; // Code [0, +inf - 1]
        let msb = (u64::BITS - 1 - x.leading_zeros()) as u64;

        UnaryCode::write_next(writer, msb, None) + writer.push_bits(x, msb)
    }

    #[inline(always)]
    fn to_encoding_type() -> EncodingType {
        EncodingType::GAMMA
    }
}

pub struct DeltaCode;

impl UniversalCode for DeltaCode {
    #[inline(always)]
    fn read_next(reader: &mut BinaryReader, _zk: Option<u64>) -> u64 {
        let msb = GammaCode::read_next(reader, None);
        ((1 << msb) | reader.read_int(msb)) - 1
    }

    #[inline(always)]
    fn write_next(writer: &mut BinaryWriter, x: u64, _zk: Option<u64>) -> u64 {
        assert!(x < u64::MAX);
        // if x < MAX_PRECOMPUTED TODO

        let x =  x + 1; // Code [0, +inf - 1]
        let msb = (u64::BITS - 1 - x.leading_zeros()) as u64;
        GammaCode::write_next(writer, msb, None) + writer.push_bits(x, msb)
    }

    #[inline(always)]
    fn to_encoding_type() -> EncodingType {
        EncodingType::DELTA
    }
}

pub struct ZetaCode;

impl UniversalCode for ZetaCode {
    #[inline(always)]
    fn read_next(reader: &mut BinaryReader, zk: Option<u64>) -> u64 {
        let zk = zk.unwrap();
        assert!(zk >= 1);

        if zk == 3 && (reader.fill >= 16 || reader.refill() >= 16) {
            let precomp = ZETAS_3[reader.current as usize >> (reader.fill - 16) & 0xFFFF];
    
            if precomp.1 != 0 {
                reader.read_bits += precomp.1 as usize;
                reader.fill -= precomp.1 as usize;

                return precomp.0 as u64;
            }
        }
    
        let unary = UnaryCode::read_next(reader, None);
        let left = 1 << (unary * zk);
        let m = reader.read_int(unary * zk + zk - 1);
        if m < left {m + left - 1} else {(m << 1) + reader.read_from_current(1) - 1}
    }

    #[inline(always)]
    fn write_next(writer: &mut BinaryWriter, x: u64, zk: Option<u64>) -> u64 {
        let zk = zk.unwrap();
        assert!(x < u64::MAX);
        assert!(zk < u64::MAX);

        let x = x + 1;
        let msb = (u64::BITS - 1 - x.leading_zeros()) as u64;
        let h = msb / zk;
        let unary = UnaryCode::write_next(writer, h, None);
        let left = 1 << (h * zk);
        unary + 
            if x - left < left 
                {writer.push_bits(x - left, h * zk + zk - 1)}
            else 
                {writer.push_bits(x, h * zk + zk)}
    }

    #[inline(always)]
    fn to_encoding_type() -> EncodingType {
        EncodingType::ZETA
    }
}

pub trait Huffman {
    fn to_encoding_type() -> EncodingType;
}

pub struct Huff;

impl Huffman for Huff {
    #[inline(always)]
    fn to_encoding_type() -> EncodingType {
        EncodingType::HUFFMAN
    }
}

/// Zuckerli encoding follows

pub const K_ZUCK: usize = 4;
pub const I_ZUCK: usize = 2;
pub const J_ZUCK: usize = 1;

#[inline(always)]
pub fn zuck_decode(token: usize, reader: &mut BinaryReader, k: usize, msb_in_token /* (i) */: usize, lsb_in_token /* (j) */: usize) -> usize { // 4 2 1
    assert!(k >= lsb_in_token + msb_in_token);
    let split_token = 1 << k; // 2^k

    if token < split_token {
        return token;
    }

    let nbits = k - (msb_in_token + lsb_in_token) +
                        ((token - split_token) >> (msb_in_token + lsb_in_token));
    let low = token & ((1 << lsb_in_token) - 1);
    let token = token >> lsb_in_token;

    let bits = reader.read_int(nbits as u64) as usize;
    (((((1 << msb_in_token) | (token & ((1 << msb_in_token) - 1)))
        << nbits) |
        bits)
        << lsb_in_token) |
        low
}

#[inline(always)]
pub fn zuck_encode(value: usize, k: usize, msb_in_token /* (i) */: usize, lsb_in_token /* (j) */: usize) -> (usize, usize, usize) {
    assert!(k >= lsb_in_token + msb_in_token);
    let split_token = 1 << k; // 2^k

    if value < split_token {
        return (value, 0, 0);
    }

    let n = (usize::BITS - 1 - value.leading_zeros()) as usize;
    let m = value - (1 << n);

    let s = split_token +                           // 2^k +
        ((n - k) << (msb_in_token + lsb_in_token)) +       // (p - k - 1) * 2^(i+j) +
        ((m >> (n -msb_in_token)) << lsb_in_token) +       // m * 2^j +
        (m & ((1 << lsb_in_token) - 1));                   // l
    let t_len = n - msb_in_token - lsb_in_token;
    let t = (value >> lsb_in_token) & ((1 << t_len) - 1);
    (s, t_len, t)
}