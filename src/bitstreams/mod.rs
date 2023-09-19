use std::rc::Rc;

use serde::{Serialize, Deserialize};

pub const GAMMA: [usize; 256] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    7<<8 | 7, 7<<8 | 7, 7<<8 | 8, 7<<8 | 8, 7<<8 | 9, 7<<8 | 9, 7<<8 | 10, 7<<8 | 10, 7<<8 | 11, 7<<8 | 11, 7<<8 | 12, 7<<8 | 12, 7<<8 | 13, 7<<8 | 13, 7<<8 | 14, 7<<8 | 14,
    5<<8 | 3, 5<<8 | 3, 5<<8 | 3, 5<<8 | 3, 5<<8 | 3, 5<<8 | 3, 5<<8 | 3, 5<<8 | 3, 5<<8 | 4, 5<<8 | 4, 5<<8 | 4, 5<<8 | 4, 5<<8 | 4, 5<<8 | 4, 5<<8 | 4, 5<<8 | 4,
    5<<8 | 5, 5<<8 | 5, 5<<8 | 5, 5<<8 | 5, 5<<8 | 5, 5<<8 | 5, 5<<8 | 5, 5<<8 | 5, 5<<8 | 6, 5<<8 | 6, 5<<8 | 6, 5<<8 | 6, 5<<8 | 6, 5<<8 | 6, 5<<8 | 6, 5<<8 | 6,
    3<<8 | 1, 3<<8 | 1, 3<<8 | 1, 3<<8 | 1, 3<<8 | 1, 3<<8 | 1, 3<<8 | 1, 3<<8 | 1, 3<<8 | 1, 3<<8 | 1, 3<<8 | 1, 3<<8 | 1, 3<<8 | 1, 3<<8 | 1, 3<<8 | 1, 3<<8 | 1,
    3<<8 | 1, 3<<8 | 1, 3<<8 | 1, 3<<8 | 1, 3<<8 | 1, 3<<8 | 1, 3<<8 | 1, 3<<8 | 1, 3<<8 | 1, 3<<8 | 1, 3<<8 | 1, 3<<8 | 1, 3<<8 | 1, 3<<8 | 1, 3<<8 | 1, 3<<8 | 1,
    3<<8 | 2, 3<<8 | 2, 3<<8 | 2, 3<<8 | 2, 3<<8 | 2, 3<<8 | 2, 3<<8 | 2, 3<<8 | 2, 3<<8 | 2, 3<<8 | 2, 3<<8 | 2, 3<<8 | 2, 3<<8 | 2, 3<<8 | 2, 3<<8 | 2, 3<<8 | 2,
    3<<8 | 2, 3<<8 | 2, 3<<8 | 2, 3<<8 | 2, 3<<8 | 2, 3<<8 | 2, 3<<8 | 2, 3<<8 | 2, 3<<8 | 2, 3<<8 | 2, 3<<8 | 2, 3<<8 | 2, 3<<8 | 2, 3<<8 | 2, 3<<8 | 2, 3<<8 | 2,
    1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0,
    1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0,
    1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0,
    1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0,
    1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0,
    1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0,
    1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0,
    1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0, 1<<8 | 0,
];

pub const ZETA_3: [usize; 256] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    7<<8 | 7, 7<<8 | 7, 7<<8 | 8, 7<<8 | 8, 7<<8 | 9, 7<<8 | 9, 7<<8 | 10, 7<<8 | 10, 7<<8 | 11, 7<<8 | 11, 7<<8 | 12, 7<<8 | 12, 7<<8 | 13, 7<<8 | 13, 7<<8 | 14, 7<<8 | 14,
    8<<8 | 15, 8<<8 | 16, 8<<8 | 17, 8<<8 | 18, 8<<8 | 19, 8<<8 | 20, 8<<8 | 21, 8<<8 | 22, 8<<8 | 23, 8<<8 | 24, 8<<8 | 25, 8<<8 | 26, 8<<8 | 27, 8<<8 | 28, 8<<8 | 29, 8<<8 | 30,
    8<<8 | 31, 8<<8 | 32, 8<<8 | 33, 8<<8 | 34, 8<<8 | 35, 8<<8 | 36, 8<<8 | 37, 8<<8 | 38, 8<<8 | 39, 8<<8 | 40, 8<<8 | 41, 8<<8 | 42, 8<<8 | 43, 8<<8 | 44, 8<<8 | 45, 8<<8 | 46,
    8<<8 | 47, 8<<8 | 48, 8<<8 | 49, 8<<8 | 50, 8<<8 | 51, 8<<8 | 52, 8<<8 | 53, 8<<8 | 54, 8<<8 | 55, 8<<8 | 56, 8<<8 | 57, 8<<8 | 58, 8<<8 | 59, 8<<8 | 60, 8<<8 | 61, 8<<8 | 62,
    3<<8 | 0, 3<<8 | 0, 3<<8 | 0, 3<<8 | 0, 3<<8 | 0, 3<<8 | 0, 3<<8 | 0, 3<<8 | 0, 3<<8 | 0, 3<<8 | 0, 3<<8 | 0, 3<<8 | 0, 3<<8 | 0, 3<<8 | 0, 3<<8 | 0, 3<<8 | 0,
    3<<8 | 0, 3<<8 | 0, 3<<8 | 0, 3<<8 | 0, 3<<8 | 0, 3<<8 | 0, 3<<8 | 0, 3<<8 | 0, 3<<8 | 0, 3<<8 | 0, 3<<8 | 0, 3<<8 | 0, 3<<8 | 0, 3<<8 | 0, 3<<8 | 0, 3<<8 | 0,
    4<<8 | 1, 4<<8 | 1, 4<<8 | 1, 4<<8 | 1, 4<<8 | 1, 4<<8 | 1, 4<<8 | 1, 4<<8 | 1, 4<<8 | 1, 4<<8 | 1, 4<<8 | 1, 4<<8 | 1, 4<<8 | 1, 4<<8 | 1, 4<<8 | 1, 4<<8 | 1,
    4<<8 | 2, 4<<8 | 2, 4<<8 | 2, 4<<8 | 2, 4<<8 | 2, 4<<8 | 2, 4<<8 | 2, 4<<8 | 2, 4<<8 | 2, 4<<8 | 2, 4<<8 | 2, 4<<8 | 2, 4<<8 | 2, 4<<8 | 2, 4<<8 | 2, 4<<8 | 2,
    4<<8 | 3, 4<<8 | 3, 4<<8 | 3, 4<<8 | 3, 4<<8 | 3, 4<<8 | 3, 4<<8 | 3, 4<<8 | 3, 4<<8 | 3, 4<<8 | 3, 4<<8 | 3, 4<<8 | 3, 4<<8 | 3, 4<<8 | 3, 4<<8 | 3, 4<<8 | 3,
    4<<8 | 4, 4<<8 | 4, 4<<8 | 4, 4<<8 | 4, 4<<8 | 4, 4<<8 | 4, 4<<8 | 4, 4<<8 | 4, 4<<8 | 4, 4<<8 | 4, 4<<8 | 4, 4<<8 | 4, 4<<8 | 4, 4<<8 | 4, 4<<8 | 4, 4<<8 | 4,
    4<<8 | 5, 4<<8 | 5, 4<<8 | 5, 4<<8 | 5, 4<<8 | 5, 4<<8 | 5, 4<<8 | 5, 4<<8 | 5, 4<<8 | 5, 4<<8 | 5, 4<<8 | 5, 4<<8 | 5, 4<<8 | 5, 4<<8 | 5, 4<<8 | 5, 4<<8 | 5,
    4<<8 | 6, 4<<8 | 6, 4<<8 | 6, 4<<8 | 6, 4<<8 | 6, 4<<8 | 6, 4<<8 | 6, 4<<8 | 6, 4<<8 | 6, 4<<8 | 6, 4<<8 | 6, 4<<8 | 6, 4<<8 | 6, 4<<8 | 6, 4<<8 | 6, 4<<8 | 6,
];

#[derive(Serialize)]
pub struct BinaryWriter {
    pub os: Box<[u8]>,
}

pub struct BinaryWriterBuilder {
    os: Vec<u8>,
    pub written_bits: usize,
    pub current: u64,
    pub free: usize,
    temp_buffer: Vec<u8>,
}

impl Default for BinaryWriterBuilder {
    fn default() -> Self {
        BinaryWriterBuilder {
            os: Vec::default(),
            written_bits: 0,
            current: 0,
            free: 8,
            temp_buffer: vec![0u8; 128]
        }
    }
}

impl BinaryWriterBuilder {
    pub fn build(mut self) -> BinaryWriter {
        if self.free < 8 {
            self.write(self.current);
        }
        
        BinaryWriter {
            os: self.os.into_boxed_slice()
        }
    }

    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn write(&mut self, b: u64) {
        self.os.push(b as u8);
    }

    #[inline(always)]
    pub fn write_in_current(&mut self, b: u64, len: u64) -> u64 {
        self.free -= len as usize;
        self.current |= (b & ((1 << len) - 1)) << self.free;

        if self.free == 0 {
            self.write(self.current);
            self.free = 8;
            self.current = 0;
        }

        self.written_bits += len as usize;
        len
    }

    #[inline(always)]
    pub fn push_bits(&mut self, x: u64, len: u64) -> u64 {
        assert!(len <= 64, "Cannot write {} bits to an integer", len);

        if len <= self.free as u64 {
            return self.write_in_current(x, len);
        }

        let mut x = x;

        let q = (len - self.free as u64) & 7;
        let blocks = (len - self.free as u64) >> 3;
        
        if q != 0 {
            self.temp_buffer[blocks as usize] = x as u8;
            x >>= q;
        }
        
        let mut i = blocks;
        while i != 0 {
            i -= 1;
            self.temp_buffer[i as usize] = x as u8;
            x >>= 8; 
        }

        self.write_in_current(x, self.free as u64);

        for i in 0..blocks {
            self.write(self.temp_buffer[i as usize] as u64);
        }

        self.written_bits += (blocks << 3) as usize;

        if q != 0 {
            self.write_in_current(self.temp_buffer[blocks as usize] as u64, q);
        }

        len
    }
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct BinaryReader {
    pub is: Rc<[u8]>,
    pub position: usize,
    pub read_bits: usize,
    pub current: u64,
    pub fill: usize,
}

impl Default for BinaryReader {
    fn default() -> Self {
        Self { 
            is: Rc::new([]), 
            position: Default::default(), 
            read_bits: Default::default(), 
            current: Default::default(), 
            fill: Default::default()
        }
    }
}

impl BinaryReader {
    pub fn new(input_stream: Rc<[u8]>) -> Self {
        BinaryReader { 
            is: input_stream, 
            ..Default::default()
        }
    }

    #[inline(always)]
    pub fn position(&mut self, pos: u64) {
        let bit_delta = ((self.position as i64) << 3) - pos as i64;
        if bit_delta >= 0 && bit_delta as usize <= self.fill {
            self.fill = bit_delta as usize;
            return;
        }

        self.fill = 0;
        self.position = pos as usize >> 3;

        let residual = pos & 7;

        if residual != 0 {
            self.current = self.read().unwrap();
            self.fill = (8 - residual) as usize;
        }
    }

    #[inline(always)]
    pub fn get_position(&self) -> usize {
        (self.position << 3) - self.fill
    }

    #[inline(always)]
    pub fn read(&mut self) -> Result<u64, ()> {
        if self.position >= self.is.len() {
            return Err(());
        }

        self.position += 1;
        Ok(self.is[self.position - 1] as u64)
    }

    #[inline(always)]
    pub fn refill(&mut self) -> usize {
        assert!(self.fill < 16);
        
        if let Ok(read) = self.read() {
            self.current = (self.current << 8) | read;
            self.fill += 8;
        }
        if let Ok(read) = self.read() {
            self.current = (self.current << 8) | read;
            self.fill += 8;
        }

        self.fill
    }

    #[inline(always)]
    pub fn read_from_current(&mut self, len: u64) -> u64 {
        if len == 0 {
            return 0;
        }

        if self.fill == 0 {
            self.current = self.read().unwrap();
            self.fill = 8;
        }

        assert!(len as usize <= self.fill);

        self.read_bits += len as usize;

        self.fill -= len as usize;
        self.current >> self.fill & ((1 << len) - 1)
    }

    #[inline(always)]
    pub fn read_int(&mut self, len: u64) -> u64 {
        assert!(len < 64);
        
        if self.fill < 16 {
            self.refill();
        }

        if len as usize <= self.fill {
            return self.read_from_current(len);
        }

        let mut len = len - self.fill as u64;
        
        let mut x = self.read_from_current(self.fill as u64);

        let mut i = len >> 3;

        while i != 0 {
            x = x << 8 | self.read().unwrap();
            i -= 1;
        }

        self.read_bits += len as usize & !7;

        len &= 7;

        (x << len) | self.read_from_current(len)
    }
}

#[cfg(test)]
mod tests;